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
@register_meta(aten.linalg_inv_ex.default)
def linalg_inv_ex_meta(A: Tensor, check_errors: bool=False):
    squareCheckInputs(A, 'linalg.inv_ex')
    checkFloatingOrComplex(A, 'linalg.inv_ex', allow_low_precision_dtypes=False)
    L = A.new_empty(A.shape)
    L.as_strided_(A.shape, make_contiguous_strides_for(A.shape, row_major=False))
    infos = A.new_empty(A.shape[:-2], dtype=torch.int32)
    return (L, infos)