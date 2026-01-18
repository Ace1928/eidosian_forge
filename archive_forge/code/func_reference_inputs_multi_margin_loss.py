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
def reference_inputs_multi_margin_loss(op_info, device, dtype, requires_grad, **kwargs):
    yield from sample_inputs_multi_margin_loss(op_info, device, dtype, requires_grad, **kwargs)
    _make_tensor = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    make_target = partial(_make_tensor, dtype=torch.long, requires_grad=False)
    make_weight = partial(_make_tensor, requires_grad=False)
    inputs = (((), make_target([], low=0, high=1)), ((S,), make_target([], low=0, high=S)), ((S,), make_target([1], low=0, high=S)), ((M, S), make_target([M], low=0, high=S)))
    ps = (1, 2)
    margins = (0, 7, -3.14)
    weights = (False, True)
    reductions = (None, 'none', 'mean', 'sum')
    for (input_shape, target), p, margin, weight, reduction in product(inputs, ps, margins, weights, reductions):
        input = _make_tensor(input_shape)
        weight_shape = [input.size(-1)] if input.ndim > 0 else [1]
        weight = make_weight(weight_shape, low=-10.0, high=10.0) if weight else None
        kwargs = {'p': p, 'margin': margin, 'weight': weight}
        if reduction is not None:
            kwargs['reduction'] = reduction
        yield SampleInput(input, args=(target,), kwargs=kwargs)