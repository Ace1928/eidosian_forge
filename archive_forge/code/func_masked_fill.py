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
@register_decomposition(aten.masked_fill)
@out_wrapper()
def masked_fill(a: TensorLikeType, mask: TensorLikeType, value: TensorOrNumberLikeType):
    python_type = utils.dtype_to_type(a.dtype)
    if isinstance(value, Number):
        value_type = type(value)
    else:
        value_ndim = value.ndim
        torch._check(value_ndim == 0, lambda: f'only supports a 0-dimensional value tensor, but got tensor with {value_ndim} dimension')
        is_cpu_scalar = a.device.type in ['cuda', 'xpu'] and value.device.type == 'cpu'
        torch._check(is_cpu_scalar or value.device == a.device, lambda: 'Expected `value` to be on same device as `a`')
        value_type = utils.dtype_to_type(value.dtype)
    if value_type is complex:
        torch._check(utils.is_weakly_lesser_type(value_type, python_type), lambda: f'could not convert to type {python_type} without overflow')
    value = _maybe_convert_to_dtype(value, a.dtype)
    r = torch.where(mask, value, a)
    return r.contiguous()