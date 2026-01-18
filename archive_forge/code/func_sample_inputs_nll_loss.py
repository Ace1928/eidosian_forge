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
def sample_inputs_nll_loss(op_info, device, dtype, requires_grad, **kwargs):
    shape = (2, 3)
    num_classes = shape[1]
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    make_weight = partial(make_tensor, num_classes, device=device, dtype=dtype, requires_grad=False)

    def make_target(shape, zeros=False):
        s = (shape[0], *shape[2:]) if len(shape) > 1 else ()
        if zeros:
            return torch.zeros(s, device=device, dtype=torch.long)
        else:
            return make_tensor(s, low=0, high=shape[1] if len(shape) > 1 else shape[0], device=device, dtype=torch.long)

    def gen_shape_kwargs():
        shapes = (shape, (num_classes,), shape + (2, 2))
        reductions = ('none', 'mean', 'sum')
        for reduction, s in product(reductions, shapes):
            yield (make_input(s), make_target(s), dict(reduction=reduction))
            yield (make_input(s), make_target(s), dict(weight=make_weight(), reduction=reduction))
            yield (make_input(s), make_target(s), dict(weight=make_weight(low=0), reduction=reduction))
            yield (make_input(s), make_target(s), dict(weight=make_weight(high=0), reduction=reduction))
            t = make_target(s)
            ignore = num_classes // 2
            if t.eq(ignore).all() and reduction == 'mean':
                t.fill_(0)
            yield (make_input(s), t, dict(ignore_index=num_classes // 2, reduction=reduction))
            yield (make_input(s), t, dict(ignore_index=num_classes // 2, reduction=reduction, weight=make_weight()))
            if reduction != 'mean':
                yield (make_input(s), make_target(s, zeros=True), dict(ignore_index=0, reduction=reduction))
    for input, target, kwargs in gen_shape_kwargs():
        yield SampleInput(input, args=(target,), kwargs=kwargs)
    target = torch.tensor([-1, 2], device=device, dtype=torch.long)
    yield SampleInput(make_input(shape), args=(target,), kwargs={'ignore_index': -1})