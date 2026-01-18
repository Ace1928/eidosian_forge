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
def linearSolveCheckInputs(self: Tensor, A: Tensor, name: str):
    torch._check(self.device == A.device, lambda: f'Expected b and A to be on the same device, but found b on {self.device} and A on {A.device} instead.')
    torch._check(self.dtype == A.dtype, lambda: f'Expected b and A to have the same dtype, but found b of type {self.dtype} and A of type {A.dtype} instead.')
    torch._check(A.size(-1) == A.size(-2), lambda: f'A must be batches of square matrices, but they are {A.size(-2)} by {A.size(-1)} matrices')
    torch._check(A.size(-1) == self.size(-2), lambda: f'Incompatible matrix sizes for {name}: each A matrix is {A.size(-1)} by {A.size(-1)} but each b matrix is {self.size(-2)} by {self.size(-1)}')