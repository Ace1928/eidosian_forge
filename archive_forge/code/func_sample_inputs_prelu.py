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
def sample_inputs_prelu(op_info, device, dtype, requires_grad, **kwargs):
    op_kwargs = op_info.sample_kwargs(device, dtype, None)[0]
    yield from sample_inputs_elementwise_unary(op_info, device, dtype, requires_grad, op_kwargs=op_kwargs)
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    cases = ((), (S,), (S, S), (S, M, S))
    for shape in cases:
        for weight in [-1.0, 0.0, 0.8, 1.0]:
            weight_tensor = torch.tensor(weight, device=device, dtype=dtype, requires_grad=requires_grad)
            yield SampleInput(make_arg(shape), args=(weight_tensor,))
        channel_size = shape[1] if len(shape) >= 2 else 1
        yield SampleInput(make_arg(shape), args=(make_arg((channel_size,)),))
    weight_tensor = torch.tensor(1.0, device=device, dtype=dtype, requires_grad=requires_grad)
    yield SampleInput(make_arg((S, S)), kwargs=dict(weight=weight_tensor))
    yield SampleInput(make_arg((S, S)), kwargs=dict(weight=make_arg((S,))))