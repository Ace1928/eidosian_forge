from functools import wraps, partial
from itertools import product, chain, islice
import itertools
import functools
import copy
import operator
import random
import unittest
import math
import enum
import torch
import numpy as np
from torch import inf, nan
from typing import Any, Dict, List, Tuple, Union, Sequence
from torch.testing import make_tensor
from torch.testing._internal.common_dtype import (
from torch.testing._internal.common_device_type import \
from torch.testing._internal.common_cuda import (
from torch.testing._internal.common_utils import (
import torch._refs as refs  # noqa: F401
import torch._refs.nn.functional
import torch._refs.special
import torch._refs.linalg
import torch._prims as prims  # noqa: F401
from torch.utils import _pytree as pytree
from packaging import version
from torch.testing._internal.opinfo.core import (  # noqa: F401
from torch.testing._internal.opinfo.refs import (  # NOQA: F401
from torch.testing._internal.opinfo.utils import (
from torch.testing._internal import opinfo
from torch.testing._internal.opinfo.definitions.linalg import (
from torch.testing._internal.opinfo.definitions.special import (
from torch.testing._internal.opinfo.definitions._masked import (
from torch.testing._internal.opinfo.definitions.sparse import (
def reference_inputs_clone_contiguous(op, device, dtype, requires_grad, **kwargs):
    yield from sample_inputs_clone_contiguous(op, device, dtype, requires_grad, **kwargs)
    shapes = ((3, 5, 6), (1, 1, 3, 5, 6), (1, 1, 3, 5, 6, 1, 1), (1, 0, 3, 5, 0, 2), (1, 0, 3, 5, 0, 0, 1, 1, 2), ())
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)
    for shape in shapes:
        yield SampleInput(make_arg(shape))
        yield SampleInput(make_arg(shape).transpose(0, -1))
        yield SampleInput(make_arg(shape, noncontiguous=True))
        yield SampleInput(make_arg(shape, noncontiguous=True).transpose(0, -1))
        yield SampleInput(make_arg(shape), kwargs={'memory_format': torch.contiguous_format})
        yield SampleInput(make_arg(shape).transpose(0, -1), kwargs={'memory_format': torch.contiguous_format})
        yield SampleInput(make_arg(shape, noncontiguous=True), kwargs={'memory_format': torch.contiguous_format})
        yield SampleInput(make_arg(shape, noncontiguous=True).transpose(0, -1), kwargs={'memory_format': torch.contiguous_format})
    strided_cases = (((5, 6, 2), (1, 1, 7), 2), ((5, 5, 4), (1, 1, 7), 2), ((5, 5, 2), (4, 5, 7), 3), ((5, 5, 2), (5, 5, 7), 3), ((5, 5, 2), (5, 5, 5), 3), ((9, 5, 2), (0, 1, 7), 3))
    for shape, strides, offset in strided_cases:
        yield SampleInput(make_arg(500).as_strided(shape, strides, offset))
        yield SampleInput(make_arg(500).as_strided(shape, strides, offset), kwargs={'memory_format': torch.contiguous_format})
    yield SampleInput(make_arg((2, 2, 2, 2)), kwargs={'memory_format': torch.channels_last})
    a = make_arg((2, 2, 2, 2)).permute(0, 3, 1, 2)
    yield SampleInput(a, kwargs={'memory_format': torch.channels_last})
    yield SampleInput(make_arg((2, 2, 2, 2, 2)), kwargs={'memory_format': torch.channels_last_3d})
    a = make_arg((2, 2, 2, 2, 2)).permute(0, 4, 1, 2, 3)
    yield SampleInput(a, kwargs={'memory_format': torch.channels_last_3d})