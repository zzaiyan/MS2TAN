import torch
import torch.nn as nn
import utils.pytorch_ssim as pssim
from utils.util_image import *
from utils.util_metrics import *
from utils.util_tiff import *
from torchvision import models, transforms

from .attention import MFE


def init_weights(net, init_type='kaiming', gain=0.02):
    from torch.nn import init

    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0.2, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError(
                    'initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)
    net.apply(init_func)


# percept = models.vgg16_bn(pretrained=False).cuda()
# state_dict = torch.load(
#     '/root/autodl-tmp/STAIR/pretrain/vgg16_bn-6c64b313.pth')
# percept.load_state_dict(state_dict)
# percept.eval()
# percept_norm = transforms.Normalize(
#     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


class MS2TAN(nn.Module):
    def __init__(
        self,
        dim_list=[256, 192, 128],
        num_frame=10,
        image_size=120,
        patch_list=[12, 10, 8],
        in_chans=7,
        out_chans=6,
        depth_list=[1, 1, 1],
        heads_list=[8, 8, 8],
        dim_head_list=[32, 24, 16],
        attn_dropout=0.0,
        ff_dropout=0.0,
        optim_input=False,
        missing_mask=True,
        enable_model=True,
        enable_conv=False,
        enable_mse=True,
        enable_struct=False,
        enable_percept=False,
    ):
        super().__init__()
        assert (enable_mse or enable_struct or enable_percept)
        self.num_block = len(dim_list)
        self.in_chans = in_chans
        self.blocks = nn.ModuleList(
            [
                MFE(
                    dim=dim_list[i],
                    num_frames=num_frame,
                    image_size=image_size,
                    patch_size=patch_list[i],
                    in_channels=in_chans,
                    out_channels=out_chans,
                    depth=depth_list[i],
                    heads=heads_list[i],
                    dim_head=dim_head_list[i],
                    attn_dropout=attn_dropout,
                    ff_dropout=ff_dropout,
                    missing_mask=(i == 0 and missing_mask),
                ) if enable_model else None
                for i in range(self.num_block)
            ]
        )
        self.out_chans = out_chans
        self.optim_input = optim_input
        self.enable_conv = enable_conv
        self.enable_model = enable_model
        self.enable_mse = enable_mse
        self.enable_struct = enable_struct
        self.enable_percept = enable_percept

        if self.enable_conv:
            # 维度变换
            in_ch, out_ch, cnum = 6, 6, 32
            self.first_conv = nn.Sequential(
                nn.Flatten(0, 1),
                nn.Conv2d(in_ch, cnum, 3, 1, 1),
                nn.BatchNorm2d(cnum),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Conv2d(cnum, 2*cnum, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Conv2d(2*cnum, 2*cnum, 1, 1, 0),
                nn.BatchNorm2d(2*cnum),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Conv2d(2*cnum, 2*cnum, 3, 1, 1),
                nn.BatchNorm2d(2*cnum),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Conv2d(2*cnum, out_ch, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Unflatten(0, (-1, num_frame)),
            )

            conv_inner_dim = 32
            self.after_conv = nn.ModuleList([nn.Sequential(
                nn.Flatten(0, 1),
                nn.Conv2d(out_chans, conv_inner_dim, 3, 1, 1),
                nn.BatchNorm2d(conv_inner_dim),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Conv2d(conv_inner_dim, conv_inner_dim, 1, 1, 0),
                nn.BatchNorm2d(conv_inner_dim),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Conv2d(conv_inner_dim, out_chans, 3, 1, 1),
                nn.Unflatten(0, (-1, num_frame)),
            ) for _ in range(self.num_block)])

        self.SSIM = pssim.SSIM()

    def one_percept_loss(self, X, y):
        b, t, c, h, w = X.shape
        flat_X = X.reshape(-1, c, h, w)
        flat_X = (flat_X - flat_X.mean()) / flat_X.std()
        flat_X = percept_norm(flat_X)
        flat_y = y.reshape(-1, c, h, w)
        flat_y = (flat_y - flat_y.mean()) / flat_y.std()
        flat_y = percept_norm(flat_y)
        emb1 = percept(flat_X)
        emb2 = percept(flat_y)
        loss = torch.mean(torch.square(emb1 - emb2))
        return loss
    
    def percept_loss(self, X, y):
        loss_1 = self.one_percept_loss(X[:,:,:3], y[:,:,:3])
        loss_2 = self.one_percept_loss(X[:,:,3:], y[:,:,3:])
        
        return loss_1 + loss_2
        

    def forward(self, X, extend_layers, y, mode="test"):
        b, t, c, h, w = X.shape
        block_out = []
        # extend_layers = X[:, :, -2:, :, :]
        obs_mask, art_mask = extend_layers
        # get mean_face
        mean_face = calc_mean_face(X, obs_mask)
        y_mean_face = calc_mean_face(y, obs_mask+art_mask)
        # expand
        obs_mask = obs_mask.expand(-1, -1, c, -1, -1)
        art_mask = art_mask.expand(-1, -1, c, -1, -1)
        # opt_X = obseaved + mean_face
        opt_X = X.clone()

        opt_X[obs_mask == 0] = mean_face[obs_mask == 0]

        opt_y = y.clone()
        opt_y[(obs_mask+art_mask) == 0] = y_mean_face[(obs_mask+art_mask) == 0]

        out = opt_X if self.optim_input else X

        if self.enable_conv:
            out = out + self.first_conv(out)

        for idx, block in enumerate(self.blocks):
            if self.enable_model:
                merge = torch.cat([out, extend_layers[0] * 2000], dim=2)
                if idx != -1:
                    out = out + block(merge, extend_layers[0])
                else:
                    out = block(merge, extend_layers[0])

            if self.enable_conv:
                out = out + self.after_conv[idx](out)

            block_out.append(out)

        raw_out = block_out[-1]
        
        if self.enable_mse:
            C_obs, C_art = 1, 3
        else:
            C_obs, C_art = 0, 0

        if mode == "train":
            loss_obs, loss_art = 0, 0
            for idx, view in enumerate(block_out):
                loss_obs += masked_rmse_cal(view, y, obs_mask)
                if idx != self.num_block-1:
                    loss_art += masked_rmse_cal(view, y, art_mask)
                else:
                    loss_art += masked_rmse_cal(view, y, art_mask) * self.num_block
            loss_obs /= self.num_block
            loss_art /= (self.num_block*2 - 1)

            raw_out[(obs_mask+art_mask) == 0] = opt_y[(obs_mask+art_mask) == 0]
            if self.enable_percept:
                loss_percept = self.percept_loss(
                    raw_out[:, :, :], opt_y[:, :, :])
            else:
                loss_percept = torch.as_tensor(0.).to(X.device)
            if self.enable_struct:
                loss_ssim = self.SSIM(
                    raw_out.reshape(-1, c, h, w), opt_y.reshape(-1, c, h, w))
            else:
                loss_ssim = torch.as_tensor(0.1).to(X.device)

            loss_all = loss_obs * C_obs + loss_art * C_art + loss_percept * 100 + (1 - loss_ssim) * 100
            
            return {
                'raw_out': raw_out,
                'loss_all': loss_all,
                'loss_obs': loss_obs,
                'loss_art': loss_art,
                'loss_ssim': loss_ssim,
                'loss_percept': loss_percept,
            }
        elif mode == 'val':
            torch.clamp_(raw_out, 0, y.max())
            loss_obs, loss_art = 0, 0
            for idx, view in enumerate(block_out):
                loss_obs += masked_rmse_cal(view, y, obs_mask)
                if idx != self.num_block-1:
                    loss_art += masked_rmse_cal(view, y, art_mask)
                else:
                    loss_art += masked_rmse_cal(view, y, art_mask) * self.num_block
            loss_obs /= self.num_block
            loss_art /= (self.num_block*2 - 1)

            cmp_out = raw_out.clone()
            cmp_out[(obs_mask+art_mask) == 0] = opt_y[(obs_mask+art_mask) == 0]
            if self.enable_percept:
                loss_percept = self.percept_loss(
                    cmp_out[:, :, :], opt_y[:, :, :])
            else:
                loss_percept = torch.as_tensor(0.).cuda()
            if self.enable_struct:
                loss_ssim = self.SSIM(
                    cmp_out.reshape(-1, c, h, w), opt_y.reshape(-1, c, h, w))
            else:
                loss_ssim = torch.as_tensor(0.1).to(X.device)

            loss_all = loss_obs * C_obs + loss_art * C_art + loss_percept * 100 + (1 - loss_ssim) * 100

            replace_out = raw_out.clone()
            replace_out[obs_mask != 0] = opt_X[obs_mask != 0]
            return {
                'raw_out': raw_out,
                'cmp_out': cmp_out,
                "replace_out": replace_out,
                "block_out": block_out,
                'loss_all': loss_all,
                'loss_obs': loss_obs,
                'loss_art': loss_art,
                'loss_ssim': loss_ssim,
                'loss_percept': loss_percept,
                'opt_X': opt_X,
                'opt_y': opt_y,
                'mean_face': mean_face,
            }
        elif mode == 'each':
            loss_obs = 0
            for view in block_out:
                loss_obs += masked_rmse_cal(view, y, obs_mask)
            loss_obs /= self.num_block
            loss_art = masked_rmse_cal(raw_out, y, art_mask)

            cmp_out = raw_out.clone()
            cmp_out[(obs_mask+art_mask) == 0] = 0
            if self.enable_percept:
                loss_percept = self.percept_loss(
                    cmp_out[:, :, :], y[:, :, :])
            else:
                loss_percept = torch.as_tensor(0.).to(X.device)
            if self.enable_struct:
                loss_ssim = self.SSIM(
                    cmp_out.reshape(-1, c, h, w), y.reshape(-1, c, h, w))
            else:
                loss_ssim = torch.as_tensor(0.1).to(X.device)

            loss_all = loss_obs * C_obs + loss_art * C_art + loss_percept * 100 + (1 - loss_ssim) * 100

            replace_out = raw_out.clone()
            replace_out[obs_mask != 0] = opt_X[obs_mask != 0]
            return {
                'hist_list': block_out,
                'raw_out': raw_out,
                'cmp_out': cmp_out,
                "replace_out": replace_out,
                "block_out": block_out,
                'loss_all': loss_all,
                'loss_obs': loss_obs,
                'loss_art': loss_art,
                'loss_ssim': loss_ssim,
                'loss_percept': loss_percept,
                'opt_X': opt_X,
                'opt_y': opt_y,
                'mean_face': mean_face,
                'y_mean_face': y_mean_face,
            }
        else:
            assert False, f'Error mode {mode}'
