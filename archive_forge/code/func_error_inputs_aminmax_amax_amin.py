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
def error_inputs_aminmax_amax_amin(op_info, device, is_ref=False, **kwargs):
    shape = (S, 0, S)
    err_msg_amax_amin = 'reduction'
    err_msg_aminmax = 'cannot compute aminmax over an empty dimension as the operation has no identity'
    if op_info.name in ['amax', 'amin', '_refs.amax', '_refs.amin']:
        yield ErrorInput(SampleInput(torch.rand(shape, device=device)), error_regex=err_msg_amax_amin)
    elif op_info.name in ['aminmax']:
        yield ErrorInput(SampleInput(torch.rand(shape, device=device)), error_regex=err_msg_aminmax)
    sizes = [1] * 65
    err_msg1 = 'only tensors with up to 64 dims are supported'
    yield ErrorInput(SampleInput(torch.randn(sizes, device=device), kwargs={'dim': -1}), error_regex=err_msg1)
    yield ErrorInput(SampleInput(torch.randn(sizes, device=device), kwargs={'dim': 64}), error_regex=err_msg1)
    if op_info.name in ['amax', 'amin', '_refs.amax', '_refs.amin']:
        dims = [(0, 0), (0, -4)]
        err_msg2 = 'in the list of dims'
        x = torch.randn(S, S, S, S, device=device)
        for dim in dims:
            yield ErrorInput(SampleInput(x, kwargs={'dim': dim}), error_regex=err_msg2)
    input5 = torch.randn(L, L, dtype=torch.float32, device=device)
    max_values = torch.empty(L, dtype=torch.float32, device=device)
    min_values = torch.empty(L, dtype=torch.double, device=device)
    illegal_values = torch.empty(L, dtype=torch.int, device=device)
    if is_ref:
        err_msg_amax_amin2 = "Attempting to cast from torch.float32 to out tensor with dtype torch.int32, but this can't be cast because it is not safe!"
    else:
        err_msg_amax_amin2 = "Expected the dtype for input and out to match, but got Float for input's dtype and Int for out's dtype."
    err_msg_aminmax2 = 'Expected out tensor to have dtype float, but got double instead'
    if op_info.name in ['amax', 'amin', '_refs.amax', '_refs.amin']:
        yield ErrorInput(SampleInput(input5, kwargs={'dim': 0, 'out': illegal_values}), error_regex=err_msg_amax_amin2)
    elif op_info.name in ['aminmax']:
        yield ErrorInput(SampleInput(input5, kwargs={'dim': 0, 'out': (max_values, min_values)}), error_regex=err_msg_aminmax2)
    err_msg3 = 'reduction'
    error_type = IndexError if 'refs' not in op_info.name else RuntimeError
    yield ErrorInput(SampleInput(torch.rand(shape, device=device), kwargs={'dim': 1}), error_type=error_type, error_regex=err_msg3)