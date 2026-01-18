from typing import Optional
import types, gc, os, time, re, platform
import torch
from torch.nn import functional as F
@MyFunction
def cuda_att_seq_fp16(self, x, sx, aa, bb, pp, ln_w, ln_b, k_mix, v_mix, r_mix, t_decay, t_first, kw, vw, rw, ow, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry, omx, orx, omy, ory):
    seq_len = x.shape[0]
    kvrx_and_y_bytes = x.numel() * 2
    k_bytes = seq_len * kw.shape[1] * 4
    v_bytes = seq_len * vw.shape[1] * 4
    r_bytes = seq_len * rw.shape[1] * 2
    buf = torch.empty((kvrx_and_y_bytes * 4 + k_bytes + v_bytes + r_bytes,), device=x.device, dtype=torch.int8)
    x_plus_out_t = torch.empty_like(x)
    xx = torch.ops.rwkv.att_seq(x, sx, ln_w, ln_b, k_mix, v_mix, r_mix, kw, vw, rw, ow, t_first, pp, aa, bb, t_decay, buf, x_plus_out_t)
    return (x_plus_out_t, xx[-1, :], aa, bb, pp)