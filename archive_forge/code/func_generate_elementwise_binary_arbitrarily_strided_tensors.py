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
def generate_elementwise_binary_arbitrarily_strided_tensors(op, *, device, dtype, requires_grad=False, exclude_zero=False):
    strided_cases = (((5, 6, 2), (1, 1, 7), 2), ((5, 5, 4), (1, 1, 7), 2), ((5, 5, 2), (4, 5, 7), 3), ((5, 5, 2), (5, 5, 7), 3), ((5, 5, 2), (5, 5, 5), 3), ((9, 5, 2), (0, 1, 7), 3))
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad, exclude_zero=exclude_zero)
    for shape, strides, offset in strided_cases:
        a = make_arg(500).as_strided(shape, strides, offset)
        b = make_arg(shape)
        yield SampleInput(a, args=(b,))