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
@register_meta([aten._foreach_pow.ScalarAndTensor])
def meta__foreach_pow_scalar_and_tensor(self, exponent):
    torch._check(isinstance(exponent, List), lambda: f'exponent must be a tensor list but got {type(exponent)}')
    return [torch.empty_like(e) for e in exponent]