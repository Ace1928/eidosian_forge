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
def sample_inputs_argwhere(op_info, device, dtype, requires_grad, **kwargs):
    yield SampleInput(torch.tensor([1, 0, 2, 0], dtype=dtype, device=device, requires_grad=requires_grad))
    mask = torch.tensor([[0, 1, 0, 1, 0], [1, 1, 1, 1, 0], [0, 0, 0, 1, 0], [1, 0, 1, 1, 0], [1, 0, 0, 1, 0]], dtype=torch.bool, device=device)
    t = make_tensor((S, S), dtype=dtype, device=device, requires_grad=requires_grad)
    t[mask] = 0
    yield SampleInput(t)
    t = make_tensor((S, S), dtype=dtype, device=device, requires_grad=requires_grad, noncontiguous=True)
    t[mask] = 0
    yield SampleInput(t)
    t = make_tensor((S, 0), dtype=dtype, device=device, requires_grad=requires_grad)
    yield SampleInput(t)
    yield SampleInput(torch.zeros((S,), dtype=dtype, device=device, requires_grad=requires_grad))
    yield SampleInput(make_tensor((), dtype=dtype, device=device, requires_grad=requires_grad))