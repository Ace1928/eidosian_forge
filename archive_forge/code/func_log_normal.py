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
@register_decomposition(aten.log_normal)
@out_wrapper()
@elementwise_type_promotion_wrapper(type_promoting_args=('self',), type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT)
def log_normal(self, mean=1, std=2, generator=None):
    assert generator is None
    torch._check(not utils.is_complex_dtype(self.dtype) and (not utils.is_integer_dtype(self.dtype)) and (not utils.is_boolean_dtype(self.dtype)), lambda: f'log_normal not implemented for {self.dtype}')
    torch._check(0 < std, lambda: f'log_normal_ expects std > 0.0, but found std={std}')
    return torch.exp(std * torch.randn_like(self) + mean)