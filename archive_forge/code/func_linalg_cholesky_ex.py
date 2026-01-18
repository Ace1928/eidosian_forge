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
@register_meta(aten.linalg_cholesky_ex.default)
def linalg_cholesky_ex(A: Tensor, upper: bool=False, check_errors: bool=False):
    squareCheckInputs(A, 'linalg.cholesky')
    checkFloatingOrComplex(A, 'linalg.cholesky')
    A_shape = A.shape
    ndim = len(A_shape)
    L_strides = make_contiguous_strides_for(A_shape, False)
    L = A.new_empty(A_shape)
    L.as_strided_(A_shape, L_strides)
    infos = A.new_empty(A_shape[0:ndim - 2], dtype=torch.int32)
    return (L, infos)