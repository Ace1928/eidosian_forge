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
def generate_elementwise_binary_broadcasting_tensors(op, *, device, dtype, requires_grad=False, exclude_zero=False):
    shapes = (((1,), ()), ((2,), ()), ((1,), (2,)), ((2, 1), (2,)), ((1, 2), (2,)), ((3, 2), (2,)), ((1, 3, 2), (2,)), ((1, 3, 2), (3, 2)), ((3, 1, 2), (3, 2)), ((2, 3, 2), ()), ((3, 1, 2), (1, 3, 2)))
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad, exclude_zero=exclude_zero)
    for shape, noncontiguous in product(shapes, [True, False]):
        shape_lhs, shape_rhs = shape
        lhs = make_arg(shape_lhs, noncontiguous=noncontiguous, **op.lhs_make_tensor_kwargs)
        rhs = make_arg(shape_rhs, noncontiguous=noncontiguous, **op.rhs_make_tensor_kwargs)
        yield SampleInput(lhs, args=(rhs,), broadcasts_input=True)