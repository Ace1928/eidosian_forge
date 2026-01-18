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
def linalg_solve_is_vector_rhs(input: Tensor, other: Tensor) -> bool:
    expected_batched_rhs_shape = input.shape[:-1]
    vector_case = other.ndim == 1 or (input.ndim - 1 == other.ndim and other.shape == expected_batched_rhs_shape)
    return vector_case