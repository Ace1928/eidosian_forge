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
def sample_inputs_embedding(op_info, device, dtype, requires_grad, **kwargs):

    def make_input(shape):
        return make_tensor(shape, device=device, dtype=dtype, requires_grad=requires_grad)

    def make_long_input(shape, *, low, high):
        return make_tensor(shape, device=device, dtype=torch.long, low=low, high=high)
    idx = make_long_input((), low=0, high=M)
    yield SampleInput(make_input((M, S)), args=(idx,))
    idx = make_long_input((S,), low=0, high=M)
    yield SampleInput(make_input((M, S)), args=(idx,))
    idx = make_long_input((S, S), low=0, high=M)
    yield SampleInput(make_input((M, S)), args=(idx,))
    if not requires_grad:
        idx = make_long_input((2, 2), low=0, high=S)
        idx[0, 0] = 2
        idx[1, 1] = 2
        yield SampleInput(make_input((S, S)), args=(idx,), kwargs={'padding_idx': 2})
        idx = make_long_input((2, 2), low=0, high=S)
        idx[0, 0] = 4
        idx[1, 1] = 4
        yield SampleInput(make_input((S, S)), args=(idx,), kwargs={'padding_idx': -1})
        idx = make_long_input((2, 2), low=0, high=S)
        weights = make_input((S, S)) * 2
        yield SampleInput(weights, args=(idx,), kwargs={'max_norm': 1.0})
        idx = make_long_input((2, 2), low=0, high=S)
        weights = make_input((S, S)) * 2
        yield SampleInput(weights, args=(idx,), kwargs={'max_norm': 1.0, 'norm_type': 1.0})
        idx = make_long_input((2, 2), low=0, high=S)
        idx[0, 0] = 1
        idx[0, 1] = 1
        weights = make_input((S, S))
        yield SampleInput(weights, args=(idx,), kwargs={'scale_grad_by_freq': True})
        idx = make_long_input((2, 2), low=0, high=S)
        weights = make_input((S, S))
        yield SampleInput(weights, args=(idx,), kwargs={'sparse': True})
        idx = make_long_input((3, 3), low=0, high=S)
        idx[0, 0] = 1
        idx[0, 1] = 1
        idx[1, 0] = 0
        weights = make_input((S, S)) * 2
        yield SampleInput(weights, args=(idx,), kwargs={'sparse': True, 'scale_grad_by_freq': True, 'padding_idx': 0, 'max_norm': 1.0})