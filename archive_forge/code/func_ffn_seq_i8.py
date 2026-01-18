from typing import Optional
import types, gc, os, time, re, platform
import torch
from torch.nn import functional as F
@MyFunction
def ffn_seq_i8(self, x, sx, ln_w, ln_b, k_mix, r_mix, kw, vw, rw, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry):
    xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
    sx = torch.cat((sx.unsqueeze(0), xx[:-1, :]))
    kx = xx * k_mix + sx * (1 - k_mix)
    rx = xx * r_mix + sx * (1 - r_mix)
    r = torch.sigmoid(self.mm8_seq(rx, rw, rmx, rrx, rmy, rry))
    vx = torch.square(torch.relu(self.mm8_seq(kx, kw, kmx, krx, kmy, kry)))
    out = r * self.mm8_seq(vx, vw, vmx, vrx, vmy, vry)
    return (x + out, xx[-1, :])