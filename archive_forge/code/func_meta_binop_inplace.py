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
@register_meta([aten.mul_.Scalar, aten.div_.Scalar, aten.mul_.Tensor, aten.div_.Tensor, aten.logical_and_.default, aten.logical_or_.default, aten.logical_xor_.default])
def meta_binop_inplace(self, other):
    if isinstance(other, torch.Tensor):
        check_inplace_broadcast(self.shape, other.shape)
    return self