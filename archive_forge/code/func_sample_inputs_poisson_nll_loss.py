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
def sample_inputs_poisson_nll_loss(op_info, device, dtype, requires_grad, **kwargs):
    _make_tensor = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    def gen_shape_kwargs():
        for s, r in _generate_sample_shape_reduction():
            for li in (True, False):
                for f in (True, False):
                    i1 = _make_tensor(s)
                    i2 = _make_tensor(s)
                    t1 = _make_tensor(s, low=0)
                    t2 = _make_tensor(s, low=0)
                    if not li:
                        i1.abs_()
                        i2.abs_()
                    t1.abs_()
                    t2.abs_()
                    yield (i1, t1, dict(log_input=li, full=f, reduction=r))
                    yield (i2, t2, dict(log_input=li, full=f, eps=random.uniform(1e-08, 0.001), reduction=r))
    for input, target, kwargs in gen_shape_kwargs():
        yield SampleInput(input, args=(target,), kwargs=kwargs)
    if dtype.is_complex:
        for d in (torch.bool, torch.int64):
            yield SampleInput(_make_tensor(dtype=dtype), args=(_make_tensor(dtype=d),))
            yield SampleInput(_make_tensor(dtype=d), args=(_make_tensor(dtype=dtype),))