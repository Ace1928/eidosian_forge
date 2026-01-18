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
def get_operator_enum(reduce_, use_new_options=False):
    if use_new_options:
        if reduce_ == 'sum':
            return 'REDUCE_ADD'
        elif reduce_ == 'prod':
            return 'REDUCE_MULTIPLY'
        elif reduce_ == 'mean':
            return 'REDUCE_MEAN'
        elif reduce_ == 'amax':
            return 'REDUCE_MAXIMUM'
        elif reduce_ == 'amin':
            return 'REDUCE_MINIMUM'
        torch._check(False, lambda: 'reduce argument must be either sum, prod, mean, amax or amin.')
        return
    else:
        if reduce_ == 'add':
            return 'REDUCE_ADD'
        elif reduce_ == 'multiply':
            return 'REDUCE_MULTIPLY'
        torch._check(False, lambda: 'reduce argument must be either add or multiply.')
        return