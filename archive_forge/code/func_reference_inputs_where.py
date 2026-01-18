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
def reference_inputs_where(op, device, dtype, requires_grad, **kwargs):
    yield from sample_inputs_where(op, device, dtype, requires_grad, **kwargs)
    make_cond = partial(make_tensor, dtype=torch.bool, device=device, requires_grad=requires_grad)
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)
    c = make_cond((10, 3), noncontiguous=True)
    a = make_arg((10, 1), noncontiguous=True)
    b = make_arg((3, 10, 3)).transpose(0, -1)
    yield SampleInput(a, args=(c, b))
    other_dtype = torch.double if dtype is not torch.double else torch.long
    c = make_cond((10, 3), noncontiguous=True)
    a = make_arg((10, 1), dtype=torch.long)
    b = make_arg((10, 1))
    yield SampleInput(a, args=(c, b))
    c = make_cond((10, 3), noncontiguous=True)
    a = make_arg((1,)).item()
    b = make_arg((1,)).item()
    yield SampleInput(a, args=(c, b))
    if dtype.is_floating_point or dtype.is_complex:
        if dtype.is_floating_point:
            nan = float('nan')
        else:
            nan = complex(float('nan'), float('nan'))
        c = make_cond((1, 10, 3))
        a = make_arg((10, 3), noncontiguous=True)
        a[2, 1] = nan
        b = make_arg((1, 3))
        b[0, 2] = nan
        yield SampleInput(a, args=(c, b))
    for scalar in (0, 0.0, 2j, False):
        yield SampleInput(scalar, args=(c, b))
        yield SampleInput(a, args=(c, scalar))