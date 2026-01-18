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
def generate_elementwise_binary_extremal_value_tensors(op, *, device, dtype, requires_grad=False):
    _float_extremals = (float('inf'), float('-inf'), float('nan'))
    l_vals = []
    r_vals = []
    if dtype.is_floating_point:
        prod = product(_float_extremals, _float_extremals)
    elif dtype.is_complex:
        complex_vals = product(_float_extremals, _float_extremals)
        complex_vals = [complex(*x) for x in complex_vals]
        prod = product(complex_vals, complex_vals)
    else:
        raise ValueError('Unsupported dtype!')
    for l, r in prod:
        l_vals.append(l)
        r_vals.append(r)
    lhs = torch.tensor(l_vals, device=device, dtype=dtype, requires_grad=requires_grad)
    rhs = torch.tensor(r_vals, device=device, dtype=dtype, requires_grad=requires_grad)
    yield SampleInput(lhs, args=(rhs,))
    nan = float('nan') if dtype.is_floating_point else complex(float('nan'), float('nan'))
    lhs = make_tensor((128, 128), device=device, dtype=dtype, requires_grad=requires_grad)
    lhs.view(-1)[::3] = nan
    rhs = make_tensor((128, 128), device=device, dtype=dtype, requires_grad=requires_grad)
    rhs.view(-1)[::3] = nan
    yield SampleInput(lhs, args=(rhs,))