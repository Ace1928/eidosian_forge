from typing import Optional
import types, gc, os, time, re, platform
import torch
from torch.nn import functional as F
@MyFunction
def cuda_att_one_v5_fp16(self, x, sx, s, ln_w, ln_b, lx_w, lx_b, k_mix, v_mix, r_mix, t_decay, t_first, kw, vw, rw, ow, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry, omx, orx, omy, ory):
    kx = torch.empty_like(x)
    vx = torch.empty_like(x)
    rx = torch.empty_like(x)
    H = t_decay.shape[0]
    S = x.shape[-1] // H
    r = torch.empty((H * S,), dtype=torch.float32, device=x.device)
    k = torch.empty((H * S,), dtype=torch.float32, device=x.device)
    v = torch.empty((H * S,), dtype=torch.float32, device=x.device)
    s1 = torch.empty((H, S, S), dtype=torch.float32, device=x.device)
    s2 = torch.empty((H, S, S), dtype=torch.float32, device=x.device)
    x_plus_out = torch.empty_like(x)
    xx = torch.ops.rwkv.att_one_v5(x, sx, s, ln_w, ln_b, lx_w, lx_b, k_mix, v_mix, r_mix, kw, kx, vw, vx, rw, rx, ow, t_first, k, t_decay, v, r, s1, x_plus_out, s2)
    return (x_plus_out, xx, s2)