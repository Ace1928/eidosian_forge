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
@register_meta([aten._foreach_addcdiv.Scalar, aten._foreach_addcmul.Scalar])
def meta__foreach_addcop_scalar(self, tensor1, tensor2, scalar=1):
    torch._check(all((isinstance(l, List) for l in [self, tensor1, tensor2])), lambda: f'All arguments must be List[Tensor], but got {type(self)}, {type(tensor1)}, and {type(tensor2)}')
    torch._check(len(self) > 0, lambda: 'input tensor list must not be empty.')
    torch._check(len(self) == len(tensor1) and len(self) == len(tensor2), lambda: 'All input tensor lists must have the same length')
    return [torch.empty_like(s) for s in self]