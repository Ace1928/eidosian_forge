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
@register_meta([aten.linalg_ldl_solve.default, aten.linalg_ldl_solve.out])
@out_wrapper()
def linalg_ldl_solve_meta(LD: Tensor, pivots: Tensor, B: Tensor, *, hermitian: bool=False) -> Tensor:
    squareCheckInputs(LD, 'torch.linalg.ldl_solve')
    checkFloatingOrComplex(LD, 'torch.linalg.ldl_solve')
    linearSolveCheckInputs(B, LD, 'torch.linalg.ldl_solve')
    torch._check(B.ndim >= 2, lambda: f'torch.linalg.ldl_solve: Expected B to have at least 2 dimensions, but it has {B.ndim} dimensions instead')
    expected_pivots_shape = LD.shape[:-1]
    torch._check(expected_pivots_shape == pivots.shape, lambda: f'torch.linalg.ldl_solve: Expected LD.shape[:-1] and pivots.shape to be the same, but got pivots with shape {pivots.shape} instead')
    torch._check(utils.is_integer_dtype(pivots.dtype), lambda: f'torch.linalg.ldl_solve: Expected pivots to be integers. Got {pivots.dtype}')
    torch._check(LD.dtype == B.dtype, lambda: f'torch.linalg.ldl_solve: LD dtype {LD.dtype} does not match b dtype {B.dtype}')
    B_broadcast_size, _ = _linalg_broadcast_batch_dims(B, LD)
    return torch.empty_strided(size=B_broadcast_size, stride=make_contiguous_strides_for(B_broadcast_size, row_major=False), dtype=B.dtype, device=B.device)