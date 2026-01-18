from typing import Optional
import types, gc, os, time, re, platform
import torch
from torch.nn import functional as F
@MyFunction
def cuda_ffn_one_fp16(self, x, sx, ln_w, ln_b, k_mix, r_mix, kw, vw, rw, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry):
    krx_bytes = x.numel() * x.element_size()
    vx_bytes = x.shape[0] * kw.shape[1] * x.element_size()
    r_bytes = x.shape[0] * rw.shape[1] * x.element_size()
    buf = torch.empty((krx_bytes * 2 + vx_bytes + r_bytes,), device=x.device, dtype=torch.int8)
    x_plus_out = torch.empty_like(x)
    xx = torch.ops.rwkv.ffn_one(x, sx, ln_w, ln_b, k_mix, r_mix, kw, vw, rw, buf, x_plus_out)
    return (x_plus_out, xx)