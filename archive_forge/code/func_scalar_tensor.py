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
def scalar_tensor(a: NumberType, *, dtype: Optional[torch.dtype]=None, layout: torch.layout=torch.strided, device: Optional[DeviceLikeType]=None, pin_memory: bool=False) -> TensorLikeType:
    utils.check_layout(layout)
    utils.check_pin_memory(pin_memory)
    dtype = dtype if dtype is not None else utils.type_to_dtype(type(a))
    device = device if device is not None else torch.device('cpu')
    return prims.scalar_tensor(a, dtype=dtype, device=device)