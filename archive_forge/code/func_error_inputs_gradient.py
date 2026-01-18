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
def error_inputs_gradient(op_info, device, **kwargs):
    for dtype in [torch.long, torch.float32, torch.complex64]:
        t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], device=device, dtype=dtype)
        dim = (1, 0)
        spacing = [0.1]
        yield ErrorInput(SampleInput(t, kwargs=dict(spacing=spacing, dim=dim, edge_order=1)), error_type=RuntimeError, error_regex='torch.gradient expected spacing to be unspecified, a scalar ')
        yield ErrorInput(SampleInput(t, kwargs=dict(edge_order=3)), error_type=RuntimeError, error_regex='torch.gradient only supports edge_order=1 and edge_order=2.')
        dim = (1, 1)
        spacing = 0.1
        yield ErrorInput(SampleInput(t, kwargs=dict(spacing=spacing, dim=dim, edge_order=1)), error_type=RuntimeError, error_regex='dim 1 appears multiple times in the list of dims')
        dim = (0, 1)
        coordinates = [torch.tensor([1, 2, 4], device='cpu'), torch.tensor([1, 2, 4], device='meta')]
        yield ErrorInput(SampleInput(t, kwargs=dict(spacing=coordinates, dim=dim, edge_order=1)), error_type=RuntimeError, error_regex='torch.gradient expected each tensor to be on the same device,')
        yield ErrorInput(SampleInput(t, kwargs=dict(dim=3)), error_type=IndexError, error_regex='')
        t = torch.tensor([[1], [2], [3]])
        yield ErrorInput(SampleInput(t, kwargs=dict(edge_order=1)), error_type=RuntimeError, error_regex='torch.gradient expected each dimension size to be at least')
        t = torch.tensor([[1, 2], [3, 4]])
        yield ErrorInput(SampleInput(t, kwargs=dict(edge_order=2)), error_type=RuntimeError, error_regex='torch.gradient expected each dimension size to be at least')