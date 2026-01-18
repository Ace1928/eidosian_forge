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
def scatter_gather_dtype_check(method_name, self, index, src_opt=None):
    if index.numel() != 0:
        torch._check(index.dtype == torch.long, lambda: f'{method_name}(): Expected dtype int64 for index')
    if src_opt is not None:
        torch._check(self.dtype == src_opt.dtype, lambda: f'{method_name}(): Expected self.dtype to be equal to src.dtype')