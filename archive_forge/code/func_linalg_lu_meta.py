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
@register_meta([aten.linalg_lu.default, aten.linalg_lu.out])
@out_wrapper('P', 'L', 'U')
def linalg_lu_meta(A: Tensor, *, pivot: bool=True) -> Tuple[Tensor, Tensor, Tensor]:
    torch._check(A.ndim >= 2, lambda: f'linalg.lu: Expected tensor with 2 or more dimensions. Got size: {A.shape} instead')
    sizes = list(A.shape)
    m = sizes[-2]
    n = sizes[-1]
    k = min(m, n)
    sizes[-1] = m
    if pivot:
        P = A.new_empty(sizes)
    else:
        P = A.new_empty([0])
    sizes[-1] = k
    L = A.new_empty(sizes)
    sizes[-2] = k
    sizes[-1] = n
    U = A.new_empty(sizes)
    return (P, L, U)