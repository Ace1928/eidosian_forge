import ctypes as ct
from functools import reduce  # Required in Python 3
import itertools
import operator
from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
from torch import Tensor
from bitsandbytes.utils import pack_dict_to_tensor, unpack_tensor_to_dict
from .cextension import COMPILED_WITH_CUDA, lib
def optimizer_update_8bit_blockwise(optimizer_name: str, g: Tensor, p: Tensor, state1: Tensor, state2: Tensor, beta1: float, beta2: float, eps: float, step: int, lr: float, qmap1: Tensor, qmap2: Tensor, absmax1: Tensor, absmax2: Tensor, weight_decay: float=0.0, gnorm_scale: float=1.0, skip_zeros=False) -> None:
    optim_func = None
    prev_device = pre_call(g.device)
    is_on_gpu([g, p, state1, state2, qmap1, qmap2, absmax1, absmax2])
    if g.dtype == torch.float32 and state1.dtype == torch.uint8:
        optim_func = str2optimizer8bit_blockwise[optimizer_name][0]
    elif g.dtype == torch.float16 and state1.dtype == torch.uint8:
        optim_func = str2optimizer8bit_blockwise[optimizer_name][1]
    elif g.dtype == torch.bfloat16 and state1.dtype == torch.uint8 and (len(str2optimizer8bit_blockwise[optimizer_name]) == 3):
        optim_func = str2optimizer8bit_blockwise[optimizer_name][2]
    else:
        raise ValueError(f'Gradient+optimizer bit data type combination not supported: grad {g.dtype}, optimizer {state1.dtype}')
    post_call(prev_device)
    is_on_gpu([p, g, state1, state2, qmap1, qmap2, absmax1, absmax2])
    prev_device = pre_call(g.device)
    optim_func(get_ptr(p), get_ptr(g), get_ptr(state1), get_ptr(state2), ct.c_float(beta1), ct.c_float(beta2), ct.c_float(eps), ct.c_int32(step), ct.c_float(lr), get_ptr(qmap1), get_ptr(qmap2), get_ptr(absmax1), get_ptr(absmax2), ct.c_float(weight_decay), ct.c_float(gnorm_scale), ct.c_bool(skip_zeros), ct.c_int32(g.numel()))
    post_call(prev_device)