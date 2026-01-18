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
def check_argmax_argmin(name, self, dim):
    if dim is not None:
        dim = maybe_wrap_dim(dim, self.dim())
        zero_numel_check_dims(self, dim, name)
    else:
        torch._check(self.numel() != 0, lambda: f'{name}: Expected reduction dim to be specified for input.numel() == 0.')