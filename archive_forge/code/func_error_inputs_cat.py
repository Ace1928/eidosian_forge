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
def error_inputs_cat(op_info, device, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=torch.float32)
    yield ErrorInput(SampleInput([make_arg((S, S)), make_arg((S, S))], kwargs={'out': make_arg((1, S)).expand((2 * S, S))}), error_regex='unsupported operation')
    yield ErrorInput(SampleInput([], kwargs={'dim': 1}), error_regex='non-empty list of Tensors')
    yield ErrorInput(SampleInput([make_arg((S, S, L, L)), make_arg((S, 0, L - 1, L))], kwargs={'dim': 1}), error_regex='Sizes of tensors must match except in dimension')
    yield ErrorInput(SampleInput([make_arg((S, 0, L - 1, L)), make_arg((S, S, L, L))], kwargs={'dim': 1}), error_regex='Sizes of tensors must match except in dimension')
    yield ErrorInput(SampleInput([make_arg((S - 1, 0)), make_arg((S, 0, L - 1, L))], kwargs={'dim': 1}), error_regex='Tensors must have same number of dimensions')
    yield ErrorInput(SampleInput([make_arg((S, 0, L - 1, L)), make_arg((S - 1, 0))], kwargs={'dim': 1}), error_regex='Tensors must have same number of dimensions')
    x = torch.zeros(0, device=device)
    y = torch.randn((4, 6), device=device)
    err_msg = 'the written-to tensor refer to a single memory location'
    yield ErrorInput(SampleInput((x, y), kwargs={'dim': 0, 'out': x}), error_regex=err_msg)
    yield ErrorInput(SampleInput((x, y), kwargs={'dim': 0, 'out': y}), error_regex=err_msg)
    z = torch.zeros((4, 6), device=device)
    yield ErrorInput(SampleInput((y, z), kwargs={'out': z[:2, :]}), error_regex=err_msg)
    if torch.device(device).type == 'cuda':
        x_cuda = make_tensor((3, 3), device=device, dtype=torch.float32)
        y_cpu = make_tensor((3, 3), device='cpu', dtype=torch.float32)
        yield ErrorInput(SampleInput((x_cuda, y_cpu)), error_regex='Expected all tensors to be on the same device')
    yield ErrorInput(SampleInput([make_arg((L, 1)), make_arg((L, 1, 1)), make_arg((L, 1, 1))]), error_regex='Tensors must have same number of dimensions')
    yield ErrorInput(SampleInput([make_arg((S, 1, M)), make_arg((S, 1, 1)), make_arg((S, M, 1))], kwargs={'dim': 1}), error_regex='Sizes of tensors must match')
    yield ErrorInput(SampleInput((make_arg((S, 1, 1)), None)), error_type=TypeError, error_regex='got None')
    yield ErrorInput(SampleInput([make_arg(()), make_arg(())]), error_regex='zero-dimensional.*cannot be concatenated')
    d = make_tensor((2, 3), device=device, dtype=torch.double)
    x = make_tensor((2, 3), device=device, dtype=torch.float32)
    yield ErrorInput(SampleInput(x, kwargs={'out': d}), error_type=TypeError, error_regex='invalid combination of arguments')