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
def generate_elementwise_binary_tensors(op, *, device, dtype, requires_grad=False, exclude_zero=False):
    shapes = ((0,), (1, 0, 3), (), (20,), (812,), (1029, 917))
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad, exclude_zero=exclude_zero)
    for shape in shapes:
        lhs = make_arg(shape, **op.lhs_make_tensor_kwargs)
        rhs = make_arg(shape, **op.rhs_make_tensor_kwargs)
        yield SampleInput(lhs, args=(rhs,))