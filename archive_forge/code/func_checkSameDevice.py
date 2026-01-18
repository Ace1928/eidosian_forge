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
def checkSameDevice(fn_name: str, result: Tensor, input: Tensor, result_name: str='result'):
    torch._check(result.device == input.device, lambda: f'{fn_name}: Expected {result_name} and input tensors to be on the same device, but got {result_name} on {result.device} and input on {input.device}')