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
@register_meta([aten.linalg_householder_product.default, aten.linalg_householder_product.out])
@out_wrapper()
def linalg_householder_product(input: Tensor, tau: Tensor) -> Tensor:
    torch._check(input.ndim >= 2, lambda: 'torch.linalg.householder_product: input must have at least 2 dimensions.')
    torch._check(input.size(-2) >= input.size(-1), lambda: 'torch.linalg.householder_product: input.shape[-2] must be greater than or equal to input.shape[-1]')
    torch._check(input.size(-1) >= tau.size(-1), lambda: 'torch.linalg.householder_product: input.shape[-1] must be greater than or equal to tau.shape[-1]')
    torch._check(input.ndim - tau.ndim == 1, lambda: f'torch.linalg.householder_product: Expected tau to have one dimension less than input, but got tau.ndim equal to {tau.ndim} and input.ndim is equal to {input.ndim}')
    if input.ndim > 2:
        expected_batch_tau_shape = input.shape[:-2]
        actual_batch_tau_shape = tau.shape[:-1]
        torch._check(actual_batch_tau_shape == expected_batch_tau_shape, lambda: f'torch.linalg.householder_product: Expected batch dimensions of tau to be equal to input.shape[:-2], but got {actual_batch_tau_shape}')
    torch._check(tau.dtype == input.dtype, lambda: f'torch.linalg.householder_product: tau dtype {tau.dtype} does not match input dtype {input.dtype}')
    checkSameDevice('torch.linalg.householder_product', tau, input, 'tau')
    return torch.empty_strided(size=input.shape, stride=make_contiguous_strides_for(input.shape, row_major=False), dtype=input.dtype, device=input.device)