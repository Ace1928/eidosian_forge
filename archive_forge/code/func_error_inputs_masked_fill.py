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
def error_inputs_masked_fill(op_info, device, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=torch.float, requires_grad=False)
    yield ErrorInput(SampleInput(make_arg((2, 2)), args=(make_arg(()) > 0, make_arg((1,)))), error_regex='only supports a 0-dimensional value tensor, but got tensor with 1 dimension')
    yield ErrorInput(SampleInput(make_arg((2, 2)), args=(make_arg(()) > 0, 1j)), error_regex='value cannot be converted to type .* without overflow')
    yield ErrorInput(SampleInput(torch.ones(2, dtype=torch.long, device=device), args=(make_arg(()) > 0, torch.tensor(1j, device=device))), error_regex='value cannot be converted to type .* without overflow')
    if torch.device(device).type == 'cuda':
        yield ErrorInput(SampleInput(torch.randn((S, S), device='cpu'), args=(torch.randn(S, S, device='cpu') > 0, torch.randn((), device='cuda'))), error_regex='to be on same device')