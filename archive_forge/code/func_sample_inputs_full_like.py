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
def sample_inputs_full_like(self, device, dtype, requires_grad, **kwargs):

    def get_val(dtype):
        return make_tensor([], dtype=dtype, device='cpu').item()
    inputs = [((), get_val(dtype), {}), ((S, S), get_val(dtype), {}), ((0, S, 0), get_val(dtype), {}), ((S,), get_val(dtype), {'dtype': dtype, 'device': device}), ((S,), get_val(torch.double), {'dtype': torch.double}), ((S,), get_val(dtype), {'device': 'cpu'}), ((S,), get_val(torch.double), {'dtype': torch.double, 'device': 'cpu'})]
    if torch.cuda.is_available():
        inputs.append(((S,), get_val(dtype), {'device': 'cuda'}))
    for shape, fill_value, kwargs in inputs:
        t = make_tensor(shape, dtype=dtype, device=device, low=None, high=None, requires_grad=requires_grad)
        yield SampleInput(t, fill_value, **kwargs)