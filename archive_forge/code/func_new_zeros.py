import builtins
import collections
import inspect
import itertools
import math
import operator
import warnings
from collections.abc import Iterable
from enum import Enum
from functools import partial, reduce, singledispatch, wraps
from typing import Any, Callable, Dict, List, Optional, overload, Sequence, Tuple, Union
import torch
import torch._prims as prims
import torch._prims_common as utils
from torch import sym_float, sym_int
from torch._prims_common import (
from torch._prims_common.wrappers import (
from torch._decomp import register_decomposition
import torch._refs._conversions
import torch._refs.fft
import torch._refs.linalg
import torch._refs.nn.functional
import torch._refs.special
@register_decomposition(aten.new_zeros)
@out_wrapper()
def new_zeros(a: TensorLikeType, size: ShapeType, *, dtype: Optional[torch.dtype]=None, layout: Optional[torch.layout]=None, device: Optional[DeviceLikeType]=None, pin_memory: bool=False, requires_grad: bool=False) -> TensorLikeType:
    dtype = a.dtype if dtype is None else dtype
    layout = a.layout if layout is None else layout
    device = a.device if device is None else device
    return torch.full(size, False if (dtype or a.dtype) == torch.bool else 0, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory, requires_grad=requires_grad)