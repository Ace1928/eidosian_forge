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
@register_meta([aten.linalg_lu_solve.default, aten.linalg_lu_solve.out])
@out_wrapper()
def linalg_lu_solve_meta(LU: Tensor, pivots: Tensor, B: Tensor, *, left: bool=True, adjoint: bool=False) -> Tensor:
    checkFloatingOrComplex(LU, 'torch.linalg.lu_solve')
    torch._check(LU.dtype == B.dtype, lambda: f'linalg.lu_solve: Expected LU and B to have the same dtype, but found LU of type {LU.dtype} and B of type {B.dtype} instead')
    torch._check(pivots.dtype == torch.int, lambda: 'linalg.lu_solve: pivots should be a Tensor of scalar type torch.int32')
    squareCheckInputs(LU, 'torch.linalg.lu_solve')
    checkInputsSolver(LU, B, left, 'linalg.lu_solve')
    torch._check(LU.size(-1) == pivots.size(-1), lambda: 'linalg.lu_solve: Number of pivots per batch should be same as the dimension of the matrix')
    torch._check(LU.shape[:-1] == pivots.shape, lambda: f'linalg.lu_solve: Expected LU.shape[:-1] and pivots.shape to be the same, but got pivots with shape {pivots.shape} instead')
    B_broadcast_size, _ = _linalg_broadcast_batch_dims(B, LU)
    result = torch.empty_strided(size=B_broadcast_size, stride=make_contiguous_strides_for(B_broadcast_size, row_major=not left), dtype=B.dtype, device=B.device)
    if result.numel() != 0 and (not left):
        if result.is_complex():
            result = result.conj()
    return result