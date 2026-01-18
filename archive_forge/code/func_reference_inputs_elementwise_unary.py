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
def reference_inputs_elementwise_unary(op, device, dtype, requires_grad, **kwargs):
    gen = partial(_reference_inputs_elementwise_unary, op, device, dtype, requires_grad, **kwargs)
    yield from gen()
    for sample in gen():
        yield sample.noncontiguous()
    yield from generate_elementwise_unary_noncontiguous_tensors(op, device=device, dtype=dtype, requires_grad=requires_grad, **kwargs)
    yield from generate_elementwise_unary_arbitrarily_strided_tensors(op, device=device, dtype=dtype, requires_grad=requires_grad, **kwargs)