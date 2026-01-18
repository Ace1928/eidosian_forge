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
def percentile_clipping(grad: Tensor, gnorm_vec: Tensor, step: int, percentile: int=5):
    """Applies percentile clipping

    grad: torch.Tensor
        The gradient tensor.
    gnorm_vec: torch.Tensor
        Vector of gradient norms. 100 elements expected.
    step: int
        The current optimiation steps (number of past gradient norms).

    """
    prev_device = pre_call(grad.device)
    is_on_gpu([grad, gnorm_vec])
    if grad.dtype == torch.float32:
        lib.cpercentile_clipping_g32(get_ptr(grad), get_ptr(gnorm_vec), ct.c_int32(step), ct.c_int32(grad.numel()))
    elif grad.dtype == torch.float16:
        lib.cpercentile_clipping_g16(get_ptr(grad), get_ptr(gnorm_vec), ct.c_int32(step), ct.c_int32(grad.numel()))
    else:
        raise ValueError(f'Gradient type {grad.dtype} not supported!')
    post_call(prev_device)
    current_gnorm = torch.sqrt(gnorm_vec[step % 100])
    vals, idx = torch.sort(gnorm_vec)
    clip_value = torch.sqrt(vals[percentile])
    gnorm_scale = 1.0
    if current_gnorm > clip_value:
        gnorm_scale = clip_value / current_gnorm
    return (current_gnorm, clip_value, gnorm_scale)