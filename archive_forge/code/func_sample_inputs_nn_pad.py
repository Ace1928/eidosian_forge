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
def sample_inputs_nn_pad(op_info, device, dtype, requires_grad, mode, **kwargs):
    assert mode in ('constant', 'reflect', 'replicate', 'circular')
    if mode in ['reflect', 'replicate']:
        cases: tuple = (((1, 3), (1, 2)), ((1, 3), (0, 1)), ((0, 3, 3), (1, 2)), ((0, 3, 3), (0, 1)), ((1, 3, 3), (1, 2)), ((1, 3, 3), (0, 1)), ((1, 3, 3), (0, 2, 0, 1)), ((0, 3, 3, 3), (0, 2, 0, 1)), ((3, 3, 5, 5), (0, 2, 0, 1)), ((3, 3, 5, 5), (1, 1, 1, 1, 1, 1)), ((1, 3, 3, 3, 3), (1, 1, 1, 1, 1, 1)), ((1, 3, 4, 4), (-1, 1, -2, 1)))
    elif mode == 'constant':
        cases = (((1, 3), (1, 2)), ((1, 3), (0, 1)), ((1, 3), (0, 2, 0, 1)), ((0, 3, 3), (1, 2)), ((0, 3, 3), (0, 1)), ((0, 3, 3), (0, 2, 0, 1)), ((0, 3, 3), (1, 1, 1, 1, 1, 1)), ((1, 3, 3), (1, 2)), ((1, 3, 3), (0, 1)), ((1, 3, 3), (0, 2, 0, 1)), ((1, 3, 3), (1, 1, 1, 1, 1, 1)), ((0, 3, 3, 3), (1, 2)), ((0, 3, 3, 3), (0, 1)), ((0, 3, 3, 3), (0, 2, 0, 1)), ((0, 3, 3, 3), (1, 1, 1, 1, 1, 1)), ((3, 3, 5, 5), (1, 2)), ((3, 3, 5, 5), (0, 1)), ((3, 3, 5, 5), (0, 2, 0, 1)), ((3, 3, 5, 5), (1, 1, 1, 1, 1, 1)), ((1, 3, 3, 3, 3), (1, 2)), ((1, 3, 3, 3, 3), (0, 1)), ((1, 3, 3, 3, 3), (0, 2, 0, 1)), ((1, 3, 3, 3, 3), (1, 1, 1, 1, 1, 1)), ((1, 3, 4, 4), (-1, 1, -2, 1)))
    elif dtype == torch.bool:
        cases = (((2, 3, 3), (1, 2)), ((1, 3, 3), (1, 2)))
    else:
        cases = (((0, 3, 3), (1, 2)), ((0, 3, 3), (0, 1)), ((1, 3, 3), (1, 2)), ((1, 3, 3), (0, 1)), ((0, 3, 3, 3), (0, 2, 0, 1)), ((3, 3, 5, 5), (0, 2, 0, 1)), ((1, 3, 3, 3, 3), (1, 1, 1, 1, 1, 1)), ((1, 3, 4, 4), (-1, 1, -2, 1)))
    make_inp = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    if mode == 'constant':
        yield SampleInput(make_inp((1, 3, 3)), args=((2, 2),))
    if mode in ['reflect', 'replicate', 'circular']:
        for shape, pad in cases:
            yield SampleInput(make_inp(shape), args=(pad, mode))
    else:
        for pad_value in (1.0, 2.0):
            for shape, pad in cases:
                yield SampleInput(make_inp(shape), args=(pad, mode, pad_value))