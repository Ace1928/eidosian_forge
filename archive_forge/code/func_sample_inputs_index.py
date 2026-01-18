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
def sample_inputs_index(op_info, device, dtype, requires_grad, reference=False, **kwargs):
    select = 'index_select' in op_info.name
    add = 'index_add' in op_info.name
    copy = 'index_copy' in op_info.name
    fill = 'index_fill' in op_info.name
    if reference:
        make_arg = partial(torch.ones, device=device, dtype=dtype, requires_grad=requires_grad)
        make_idx = partial(torch.zeros, device=device, dtype=torch.int64)
    else:
        make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
        if copy or add:
            make_idx = partial(torch.randperm, device=device, dtype=torch.int64)
        else:

            def make_idx(n):
                return make_tensor((n,), device=device, dtype=torch.int64, low=0, high=n)
    shapes = [(), (1,), (S, S)]
    if add:
        if dtype == torch.bool:
            alphas = (True, False)
        else:
            alphas = (-1, 0, 2)
    else:
        alphas = (None,)
    if fill:
        values = (make_arg((1,)).item(), make_arg(()))
    else:
        values = (None,)
    for shape, alpha, value in product(shapes, alphas, values):
        t = make_arg(shape)
        args = []
        dim = -1 if t.ndim == 2 else 0
        args.append(dim)
        idx = make_idx(t.shape[dim] if t.ndim != 0 else 1)
        args.append(idx)
        if copy or add:
            args.append(make_arg(shape))
        elif fill:
            args.append(value)
        args = tuple(args)
        kwargs = {} if alpha is None else {'alpha': alpha}
        yield SampleInput(t, args=args, kwargs=kwargs)