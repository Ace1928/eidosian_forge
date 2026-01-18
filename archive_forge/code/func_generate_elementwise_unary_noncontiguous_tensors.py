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
def generate_elementwise_unary_noncontiguous_tensors(op, *, device, dtype, requires_grad=False):
    make_arg = partial(_make_unary_elementwise_tensor, op=op, device=device, dtype=dtype, requires_grad=requires_grad)
    t = make_arg((1026,), noncontiguous=True)
    yield SampleInput(t, kwargs=op.sample_kwargs(device, dtype, t)[0])
    t = make_arg((1024, 1024)).T
    yield SampleInput(t, kwargs=op.sample_kwargs(device, dtype, t)[0])
    shapes = ((1, 3), (1, 7), (5, 7))
    for shape in shapes:
        t = make_arg(shape)
        t_non_contig = t.expand(3, -1, -1)
        yield SampleInput(t_non_contig, kwargs=op.sample_kwargs(device, dtype, t_non_contig)[0])