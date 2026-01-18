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
def reference_inputs_elementwise_binary(op, device, dtype, requires_grad, **kwargs):
    if hasattr(op, 'rhs_make_tensor_kwargs'):
        exclude_zero = op.rhs_make_tensor_kwargs.get('exclude_zero', False)
    gen = partial(_reference_inputs_elementwise_binary, op, device, dtype, requires_grad, exclude_zero, **kwargs)
    yield from gen()
    for sample in gen():
        yield sample.noncontiguous()
    yield from generate_elementwise_binary_noncontiguous_tensors(op, device=device, dtype=dtype, requires_grad=requires_grad, exclude_zero=exclude_zero)
    yield from generate_elementwise_binary_arbitrarily_strided_tensors(op, device=device, dtype=dtype, requires_grad=requires_grad, exclude_zero=exclude_zero)