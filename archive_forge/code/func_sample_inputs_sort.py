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
def sample_inputs_sort(op_info, device, dtype, requires_grad, **kwargs):

    def small_3d_unique():
        res = torch.randperm(S * S * S, dtype=torch.int64, device=device).view(S, S, S)
        res = res.to(dtype).requires_grad_(requires_grad)
        return res

    def large_1d_unique():
        res = torch.randperm(L * L * L, dtype=torch.int64, device=device)
        res = res.to(dtype).requires_grad_(requires_grad)
        return res
    yield SampleInput(large_1d_unique())
    dims = range(-3, 3)
    flag = [True, False]
    for dim, descending, stable in product(dims, flag, flag):
        yield SampleInput(small_3d_unique(), dim, descending)
        if torch.device(device).type == 'cpu':
            yield SampleInput(small_3d_unique(), dim=dim, descending=descending, stable=stable)
    tensor_opt = dict(dtype=dtype, device=device, requires_grad=requires_grad)
    yield SampleInput(torch.tensor(1, **tensor_opt))
    yield SampleInput(torch.tensor(1, **tensor_opt), 0)
    yield SampleInput(torch.tensor(1, **tensor_opt), 0, True)
    yield SampleInput(torch.tensor((), **tensor_opt))
    yield SampleInput(torch.tensor((), **tensor_opt), 0)
    yield SampleInput(torch.tensor((), **tensor_opt), 0, True)
    yield SampleInput(small_3d_unique(), stable=True)
    yield SampleInput(small_3d_unique(), dim=0, stable=True)
    yield SampleInput(small_3d_unique(), dim=0, descending=True, stable=True)