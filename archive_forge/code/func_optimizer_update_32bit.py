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
def optimizer_update_32bit(optimizer_name: str, g: Tensor, p: Tensor, state1: Tensor, beta1: float, eps: float, step: int, lr: float, state2: Optional[torch.Tensor]=None, beta2: float=0.0, weight_decay: float=0.0, gnorm_scale: float=1.0, unorm_vec: Optional[torch.Tensor]=None, max_unorm: float=0.0, skip_zeros=False) -> None:
    """
    Performs an inplace optimizer update with one or two optimizer states.

    Universal optimizer update for 32-bit state and 32/16-bit gradients/weights.

    Parameters
    ----------
    optimizer_name : str
        The name of the optimizer: {adam}.
    g : torch.Tensor
        Gradient tensor.
    p : torch.Tensor
        Parameter tensor.
    state1 : torch.Tensor
        Optimizer state 1.
    beta1 : float
        Optimizer beta1.
    eps : float
        Optimizer epsilon.
    weight_decay : float
        Weight decay.
    step : int
        Current optimizer step.
    lr : float
        The learning rate.
    state2 : torch.Tensor
        Optimizer state 2.
    beta2 : float
        Optimizer beta2.
    gnorm_scale : float
        The factor to rescale the gradient to the max clip value.
    unorm_vec : torch.Tensor
        The tensor for the update norm.
    max_unorm : float
        The maximum update norm relative to the weight norm.
    skip_zeros : bool
        Whether to skip zero-valued gradients or not (default: False).
    """
    param_norm = 0.0
    if max_unorm > 0.0:
        param_norm = torch.norm(p.data.float())
    optim_func = None
    if g.dtype == torch.float32:
        optim_func = str2optimizer32bit[optimizer_name][0]
    elif g.dtype == torch.float16:
        optim_func = str2optimizer32bit[optimizer_name][1]
    elif g.dtype == torch.bfloat16 and len(str2optimizer32bit[optimizer_name]) == 3:
        optim_func = str2optimizer32bit[optimizer_name][2]
    else:
        raise ValueError(f'Gradient+optimizer bit data type combination not supported: grad {g.dtype}, optimizer {state1.dtype}')
    is_on_gpu([g, p, state1, state2, unorm_vec])
    prev_device = pre_call(g.device)
    optim_func(get_ptr(g), get_ptr(p), get_ptr(state1), get_ptr(state2), get_ptr(unorm_vec), ct.c_float(max_unorm), ct.c_float(param_norm), ct.c_float(beta1), ct.c_float(beta2), ct.c_float(eps), ct.c_float(weight_decay), ct.c_int32(step), ct.c_float(lr), ct.c_float(gnorm_scale), ct.c_bool(skip_zeros), ct.c_int32(g.numel()))
    post_call(prev_device)