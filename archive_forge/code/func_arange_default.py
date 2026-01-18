import functools
import numbers
import operator
import sys
from enum import Enum
from functools import partial, reduce
from itertools import chain, product
from typing import Callable, cast, Iterable, List, Optional, Tuple, Union
import torch
import torch._prims as prims
import torch._prims_common as utils
import torch.nn.functional as F
from torch import sym_float, sym_int, Tensor
from torch._decomp import register_decomposition
from torch._higher_order_ops.out_dtype import out_dtype
from torch._prims_common import IntLike, NumberType, TensorLike, TensorSequenceType
from torch._prims_common.wrappers import (
from torch.utils import _pytree as pytree
from torch.utils._pytree import tree_map
@register_decomposition([aten.arange.default, aten.arange.out])
@out_wrapper()
def arange_default(end: NumberType, *, dtype: Optional[torch.dtype]=None, layout: torch.layout=torch.strided, device: Optional[torch.device]=None, pin_memory: bool=False):
    return aten.arange.start_step(0, end, 1, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory)