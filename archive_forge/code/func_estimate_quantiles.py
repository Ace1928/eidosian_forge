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
def estimate_quantiles(A: Tensor, out: Optional[torch.Tensor]=None, offset: float=1 / 512, num_quantiles=256) -> Tensor:
    """
    Estimates 256 equidistant quantiles on the input tensor eCDF.

    Uses SRAM-Quantiles algorithm to quickly estimate 256 equidistant quantiles
    via the eCDF of the input tensor `A`. This is a fast but approximate algorithm
    and the extreme quantiles close to 0 and 1 have high variance / large estimation
    errors. These large errors can be avoided by using the offset variable which trims
    the distribution. The default offset value of 1/512 ensures minimum entropy encoding -- it
    trims 1/512 = 0.2% from each side of the distrivution. An offset value of 0.01 to 0.02
    usually has a much lower error but is not a minimum entropy encoding. Given an offset
    of 0.02 equidistance points in the range [0.02, 0.98] are used for the quantiles.

    Parameters
    ----------
    A : torch.Tensor
        The input tensor. Any shape.
    out : torch.Tensor
        Tensor with the 256 estimated quantiles.
    offset : float
        The offset for the first and last quantile from 0 and 1. Default: 1/(2*num_quantiles)
    num_quantiles : int
        The number of equally spaced quantiles.

    Returns
    -------
    torch.Tensor:
        The 256 quantiles in float32 datatype.
    """
    if A.numel() < 256:
        raise NotImplementedError(f'Quantile estimation needs at least 256 values in the Tensor, but Tensor had only {A.numel()} values.')
    if num_quantiles > 256:
        raise NotImplementedError(f'Currently only a maximum of 256 equally spaced quantiles are supported, but the argument num_quantiles={num_quantiles}')
    if num_quantiles < 256 and offset == 1 / 512:
        offset = 1 / (2 * num_quantiles)
    if out is None:
        out = torch.zeros((256,), dtype=torch.float32, device=A.device)
    is_on_gpu([A, out])
    device = pre_call(A.device)
    if A.dtype == torch.float32:
        lib.cestimate_quantiles_fp32(get_ptr(A), get_ptr(out), ct.c_float(offset), ct.c_int(A.numel()))
    elif A.dtype == torch.float16:
        lib.cestimate_quantiles_fp16(get_ptr(A), get_ptr(out), ct.c_float(offset), ct.c_int(A.numel()))
    else:
        raise NotImplementedError(f'Not supported data type {A.dtype}')
    post_call(device)
    if num_quantiles < 256:
        step = round(256 / num_quantiles)
        idx = torch.linspace(0, 255, num_quantiles).long().to(A.device)
        out = out[idx]
    return out