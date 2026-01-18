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
def error_inputs_narrow_narrow_copy(op_info, device, *, is_narrow, is_ref):
    make_arg = partial(make_tensor, device=device, dtype=torch.float32)
    yield ErrorInput(SampleInput(make_arg(()), 0, 0, 1), error_type=RuntimeError, error_regex='narrow\\(\\) cannot be applied to a 0-dim tensor\\.')
    if not is_narrow and (not is_ref) and (torch.device(device).type == 'cpu'):
        yield ErrorInput(SampleInput(make_arg((M, S, L)), 3, 0, 0), error_type=RuntimeError, error_regex='Expected dim < static_cast<int64_t>\\(self_sizes.size\\(\\)\\) to be true, but got false\\.')
    else:
        yield ErrorInput(SampleInput(make_arg((M, S, L)), 3, 0, 0), error_type=IndexError, error_regex='Dimension out of range \\(expected to be in range of \\[-3, 2\\], but got 3\\)')
    yield ErrorInput(SampleInput(make_arg((L, S, M)), -4, 0, 0), error_type=IndexError, error_regex='Dimension out of range \\(expected to be in range of \\[-3, 2\\], but got -4\\)')
    yield ErrorInput(SampleInput(make_arg((L, M, S)), 1, M + 1, 0), error_type=IndexError, error_regex='start out of range \\(expected to be in range of \\[-10, 10\\], but got 11\\)')
    yield ErrorInput(SampleInput(make_arg((L, M, S)), 1, -M - 1, 0), error_type=IndexError, error_regex='start out of range \\(expected to be in range of \\[-10, 10\\], but got -11\\)')
    yield ErrorInput(SampleInput(make_arg((S, L, M)), 2, 0, M + 1), error_type=RuntimeError, error_regex='start \\(0\\) \\+ length \\(11\\) exceeds dimension size \\(10\\)\\.')
    if not is_narrow and (not is_ref) and (torch.device(device).type == 'cpu'):
        yield ErrorInput(SampleInput(make_arg((M,)), 0, 0, -1), error_type=RuntimeError, error_regex='start \\(0\\) \\+ length \\(-1\\) exceeds dimension size \\(10\\)\\.')
    else:
        yield ErrorInput(SampleInput(make_arg((M,)), 0, 0, -1), error_type=RuntimeError, error_regex='narrow\\(\\): length must be non-negative\\.')
    if is_narrow:
        yield ErrorInput(SampleInput(make_arg((L, M, S)), 1, make_arg(S, dtype=torch.int), 2), error_type=RuntimeError, error_regex='start must be an 0-dim integral Tensor\\.')
        yield ErrorInput(SampleInput(make_arg((L, M, S)), -3, make_arg((), dtype=torch.bool), 3), error_type=RuntimeError, error_regex='start must be an 0-dim integral Tensor\\.')