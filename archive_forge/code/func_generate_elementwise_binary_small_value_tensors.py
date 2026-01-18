import collections
import collections.abc
import math
import operator
import unittest
from dataclasses import asdict, dataclass
from enum import Enum
from functools import partial
from itertools import product
from typing import Any, Callable, Iterable, List, Optional, Tuple
from torchgen.utils import dataclass_repr
import torch
from torch.testing import make_tensor
from torch.testing._internal.common_device_type import (
from torch.testing._internal.common_dtype import (
from torch.testing._internal.common_utils import (
from torch.testing._internal.opinfo import utils
def generate_elementwise_binary_small_value_tensors(op, *, device, dtype, requires_grad=False, exclude_zero=None):
    if exclude_zero is None:
        if hasattr(op, 'rhs_make_tensor_kwargs'):
            exclude_zero = op.rhs_make_tensor_kwargs.get('exclude_zero', False)
    _unsigned_int_vals = (0, 1, 55, 127, 128, 190, 210, 220, 254)
    _int_vals = (0, -1, 1, -55, 55, -127, 127, -128)
    _float_vals = (0.0, -0.0, -0.001, 0.001, -0.25, 0.25, -1.0, 1.0, -math.pi / 2, math.pi / 2, -math.pi + 1e-05, math.pi - 1e-05, -math.pi, math.pi, -math.pi - 1e-05, math.pi + 1e-05)
    l_vals = []
    r_vals = []
    if dtype.is_floating_point:
        prod = product(_float_vals, _float_vals)
    elif dtype.is_complex:
        complex_vals = product(_float_vals, _float_vals)
        complex_vals = [complex(*x) for x in complex_vals]
        prod = product(complex_vals, complex_vals)
    elif dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
        prod = product(_int_vals, _int_vals)
    elif dtype is torch.uint8:
        prod = product(_unsigned_int_vals, _unsigned_int_vals)
    else:
        raise ValueError('Unsupported dtype!')
    for l, r in prod:
        l_vals.append(l)
        if r == 0 and exclude_zero:
            r_vals.append(1)
        else:
            r_vals.append(r)
    lhs = torch.tensor(l_vals, device=device, dtype=dtype, requires_grad=requires_grad)
    rhs = torch.tensor(r_vals, device=device, dtype=dtype, requires_grad=requires_grad)
    yield SampleInput(lhs, args=(rhs,))