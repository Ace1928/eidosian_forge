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
def squareCheckInputs(self: Tensor, f_name: str):
    assert self.dim() >= 2, f'{f_name}: The input tensor must have at least 2 dimensions.'
    assert self.size(-1) == self.size(-2), f'{f_name}: A must be batches of square matrices, but they are {self.size(-2)} by {self.size(-1)} matrices'