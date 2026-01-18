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
def sample_inputs_scatter_reduce(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)
    gather = partial(gather_variable, device=device)
    zero = torch.tensor(0, dtype=torch.long, device=device)
    test_cases = (((M, S), 0, gather((S, S), 1, M), (S, S)), ((M, S), 1, gather((S, S), 0, S), (S, S)), ((M, S), -1, gather((S, S), 0, S), (S, S)), ((M, S), 0, gather((M, S // 2), 1, M), (M, S // 2)), ((M, S), 1, gather((M, S // 2), 0, S), (M, S // 2)), ((M, S), -1, gather((M, S // 2), 0, S), (M, S // 2)), ((), 0, zero.clone().detach(), ()))
    reduce = op_info.variant_test_name
    for (inp_shape, dim, index, src_shape), include_self in product(test_cases, [False, True, False]):
        yield SampleInput(make_arg(inp_shape), args=(dim, index, make_arg(src_shape), reduce), kwargs={'include_self': include_self})
    if requires_grad and reduce == 'prod':
        input = torch.tensor([[0, 13], [0, 17], [0, 19]], dtype=dtype, device=device, requires_grad=requires_grad)
        src = torch.tensor([[0, 1, 2, 3], [0, 4, 0, 1], [2, 3, 5, 6]], dtype=dtype, device=device, requires_grad=requires_grad)
        idx = torch.tensor([[1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]], dtype=torch.long, device=device)
        yield SampleInput(input, args=(1, idx, src, reduce), kwargs={'include_self': True})