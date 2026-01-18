from typing import Optional
import types, gc, os, time, re, platform
import torch
from torch.nn import functional as F
@MyFunction
def ffn_one_i8(self, x, sx, ln_w, ln_b, k_mix, r_mix, kw, vw, rw, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry):
    xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
    kx = xx * k_mix + sx * (1 - k_mix)
    rx = xx * r_mix + sx * (1 - r_mix)
    r = torch.sigmoid(self.mm8_one(rx, rw, rmx, rrx, rmy, rry))
    vx = torch.square(torch.relu(self.mm8_one(kx, kw, kmx, krx, kmy, kry)))
    out = r * self.mm8_one(vx, vw, vmx, vrx, vmy, vry)
    return (x + out, xx)