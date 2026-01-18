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
def sample_inputs_take(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)
    make_idx = partial(make_tensor, low=0, dtype=torch.int64, device=device, requires_grad=False)
    S = 3
    index = make_idx((S,), high=S * S)
    for idx in (index, -index - 1):
        yield SampleInput(input=make_arg((S, S)), args=(idx,))
    scalar_sizes = [(), (1,)]
    src_gen = (make_arg(size) for size in scalar_sizes)
    idx_gen = (make_idx(size, high=1) for size in scalar_sizes)
    for src, idx in product(src_gen, idx_gen):
        yield SampleInput(input=src.clone().requires_grad_(requires_grad), args=(idx.clone(),))
    src_sizes = [(0,), (), (1,), (3, 2)]
    src_gen = (make_arg(size) for size in src_sizes)
    idx = make_idx((0,), high=1)
    for src in src_gen:
        yield SampleInput(input=src.clone().requires_grad_(requires_grad), args=(idx.clone(),))