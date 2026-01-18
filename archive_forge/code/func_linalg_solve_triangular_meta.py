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
@register_meta([aten.linalg_solve_triangular.default, aten.linalg_solve_triangular.out])
def linalg_solve_triangular_meta(A: Tensor, B: Tensor, *, upper: bool, left: bool=True, unitriangular: bool=False, out: Optional[Tensor]=None) -> Tensor:
    if out is None:
        out = A.new_empty([0])
    assert isinstance(out, TensorLike)
    checkInputsSolver(A, B, left, 'linalg.solve_triangular')
    B_, A_ = _linalg_broadcast_batch_dims_name(B, A, None)
    avoid_copy_A = A_.transpose(-2, -1).is_contiguous() and A_.is_conj()
    if avoid_copy_A:
        out = _maybe_resize_out(out, B_.shape)
    elif _resize_output_check(out, B_.shape):
        out.resize_(B_.transpose(-2, -1).shape)
        out.transpose_(-2, -1)
    return out