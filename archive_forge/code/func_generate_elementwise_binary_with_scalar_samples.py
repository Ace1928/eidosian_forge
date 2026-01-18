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
def generate_elementwise_binary_with_scalar_samples(op, *, device, dtype, requires_grad=False):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    shapes = ((), (3,), (5, 3), (0, 1, 3), (1, 5))
    if op.supports_rhs_python_scalar:
        for shape in shapes:
            lhs = make_arg(shape, **op.lhs_make_tensor_kwargs)
            rhs = make_arg(shape, **op.rhs_make_tensor_kwargs)
            lhs_scalar = make_arg((), **op.lhs_make_tensor_kwargs).item()
            rhs_scalar = make_arg((), **op.rhs_make_tensor_kwargs).item()
            yield SampleInput(lhs, args=(rhs_scalar,))
        if op.supports_one_python_scalar:
            yield SampleInput(lhs_scalar, args=(rhs,))
    if op.supports_two_python_scalars:
        lhs_scalar = make_arg((), **op.lhs_make_tensor_kwargs).item()
        rhs_scalar = make_arg((), **op.rhs_make_tensor_kwargs).item()
        yield SampleInput(lhs_scalar, args=(rhs_scalar,))