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
@register_meta(aten.lu_unpack)
@out_wrapper('P', 'L', 'U')
def lu_unpack_meta(LU: Tensor, pivots: Tensor, unpack_data: bool=True, unpack_pivots: bool=True) -> Tuple[Tensor, Tensor, Tensor]:
    torch._check(LU.ndim >= 2, lambda: f'torch.lu_unpack: Expected tensor with 2 or more dimensions. Got size: {LU.shape} instead')
    if unpack_pivots:
        torch._check(pivots.dtype == torch.int32, lambda: 'torch.lu_unpack: LU_pivots is expected to be a contiguous tensor of torch.int32 dtype.\nNote: this function is intended to be used with the output produced by torch.linalg.lu_factor')
    sizes = list(LU.shape)
    m = sizes[-2]
    n = sizes[-1]
    k = min(m, n)
    sizes[-1] = m
    if unpack_pivots:
        P = LU.new_empty(sizes)
    else:
        P = LU.new_empty([0])
    if unpack_data:
        sizes[-1] = k
        L = LU.new_empty(sizes)
        sizes[-2] = k
        sizes[-1] = n
        U = LU.new_empty(sizes)
    else:
        L = LU.new_empty([0])
        U = LU.new_empty([0])
    return (P, L, U)