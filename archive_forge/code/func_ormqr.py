import math
from enum import Enum
from functools import partial
from typing import List, Optional, Sequence, Tuple, Union
import torch
import torch._prims_common as utils
from torch import SymBool, SymFloat, Tensor
from torch._decomp import (
from torch._ops import OpOverload
from torch._prims import _prim_elementwise_meta, ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND
from torch._prims_common import (
from torch._prims_common.wrappers import (
from torch._refs import _broadcast_shapes, _maybe_broadcast
from torch.utils import _pytree as pytree
import torch._refs
import torch._refs.nn.functional
import torch._refs.special
@register_meta(aten.ormqr)
@out_wrapper()
def ormqr(input: Tensor, tau: Tensor, other: Tensor, left: bool=True, transpose: bool=False) -> Tensor:
    torch._check(input.ndim >= 2, lambda: 'torch.ormqr: input must have at least 2 dimensions.')
    torch._check(other.ndim >= 2, lambda: 'torch.ormqr: other must have at least 2 dimensions.')
    left_size_condition = -2 if left else -1
    torch._check(other.shape[left_size_condition] >= tau.shape[-1], lambda: f'torch.ormqr: other.shape[{left_size_condition}] must be greater than or equal to tau.shape[-1]')
    torch._check(other.shape[left_size_condition] == input.shape[-2], lambda: f'torch.ormqr: other.shape[{left_size_condition}] must be equal to input.shape[-2]')
    torch._check(tau.shape[-1] <= input.shape[-1], lambda: 'torch.ormqr: tau.shape[-1] must be less than or equal to input.shape[-1]')
    torch._check(input.ndim - tau.ndim == 1, lambda: f'torch.ormqr: Expected tau to have one dimension less than input, but got tau.ndim equal to {tau.ndim} and input.ndim is equal to {input.ndim}')
    torch._check(input.ndim == other.ndim, lambda: f'torch.ormqr: Expected other to have the same number of dimensions as input, but got other.ndim equal to {other.ndim} and input.ndim is equal to {input.ndim}')
    if input.ndim > 2:
        expected_batch_shape = input.shape[:-2]
        actual_batch_tau_shape = tau.shape[:-1]
        torch._check(actual_batch_tau_shape == expected_batch_shape, lambda: f'torch.ormqr: Expected batch dimensions of tau to be equal to input.shape[:-2], but got {actual_batch_tau_shape}')
        actual_batch_other_shape = other.shape[:-2]
        torch._check(actual_batch_other_shape == expected_batch_shape, lambda: f'torch.ormqr: Expected batch dimensions of other to be equal to input.shape[:-2], but got {actual_batch_other_shape}')
    torch._check(tau.dtype == input.dtype, lambda: f'torch.ormqr: Expected input and tau to have the same dtype, but input has dtype {input.dtype} and tau has dtype {tau.dtype}')
    torch._check(other.dtype == input.dtype, lambda: f'torch.ormqr: Expected input and other to have the same dtype, but input has dtype {input.dtype} and other has dtype {other.dtype}')
    checkSameDevice('torch.ormqr', tau, input, 'tau')
    checkSameDevice('torch.ormqr', other, input, 'other')
    return torch.empty_strided(size=other.shape, stride=make_contiguous_strides_for(other.shape, row_major=False), dtype=other.dtype, device=other.device)