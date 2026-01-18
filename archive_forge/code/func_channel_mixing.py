import numpy as np
import types, torch
from torch.nn import functional as F
from tokenizers import Tokenizer
@torch.jit.script_method
def channel_mixing(self, x, state, i: int, time_mix_k, time_mix_r, kw, vw, rw):
    xk = x * time_mix_k + state[5 * i + 0] * (1 - time_mix_k)
    xr = x * time_mix_r + state[5 * i + 0] * (1 - time_mix_r)
    state[5 * i + 0] = x
    r = torch.sigmoid(rw @ xr)
    k = torch.square(torch.relu(kw @ xk))
    return r * (vw @ k)