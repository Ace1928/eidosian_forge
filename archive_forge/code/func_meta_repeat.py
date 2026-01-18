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
@register_meta(aten.repeat.default)
def meta_repeat(self, repeats):
    torch._check(len(repeats) >= self.dim(), lambda: 'Number of dimensions of repeat dims can not be smaller than number of dimensions of tensor')
    num_new_dimensions = len(repeats) - self.dim()
    padded_size = (1,) * num_new_dimensions + tuple(self.shape)
    target_size = [padded_size[i] * repeats[i] for i in range(len(repeats))]
    return self.new_empty(target_size)