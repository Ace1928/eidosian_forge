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
def sample_inputs_index_reduce(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    def make_idx(n, m):
        return make_tensor((n,), device=device, dtype=torch.int64, low=0, high=m)
    shapes = [((), ()), ((1,), (1,)), ((S, S), (S, M)), ((S, S, S), (S, M, S))]
    include_selfs = (True, False)
    reduces = ('prod', 'mean', 'amin', 'amax')
    for shape, include_self, reduce in product(shapes, include_selfs, reduces):
        self_shape, src_shape = shape
        dim = 1 if len(self_shape) >= 2 else 0
        idx = make_idx(src_shape[dim] if len(src_shape) != 0 else 1, self_shape[dim] if len(self_shape) != 0 else 1)
        args = (dim, idx, make_arg(src_shape), reduce)
        yield SampleInput(make_arg(self_shape), args=args, kwargs={'include_self': include_self})
    if requires_grad:
        input = torch.tensor([[0, 13], [0, 0], [15, 19]], dtype=dtype, device=device, requires_grad=requires_grad)
        src = torch.tensor([[2, 0], [0, 0], [2, 3], [2, 2]], dtype=dtype, device=device, requires_grad=requires_grad)
        idx = torch.tensor([0, 1, 2, 0], dtype=torch.long, device=device)
        yield SampleInput(input, args=(0, idx, src, 'prod'), kwargs={'include_self': True})