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
def sample_inputs_new_fns(self, device, dtype, requires_grad, *, is_strided=False, **kwargs):
    inputs = [((), (), (), {}), ((S, S), (2, 0), (3, 4), {}), ((0, S, 0), (3, 2, 2), (1, 2, 3), {}), ((S,), (2, 3), (7, 8), {'dtype': dtype, 'device': device}), ((S,), (10,), (S,), {'dtype': torch.double}), ((S,), (1, 1, 12), (S, L, M), {'device': 'cpu'}), ((S,), (2, 2, 2), (L, M, S), {'dtype': torch.double, 'device': 'cpu'})]
    if torch.cuda.is_available():
        inputs.append(((S,), (7, 2), (3, 4), {'device': 'cuda'}))
    for input_shape, output_shape, strides, kwargs in inputs:
        t = make_tensor(input_shape, dtype=dtype, device=device, low=None, high=None, requires_grad=requires_grad)
        if is_strided:
            yield SampleInput(t, output_shape, strides, **kwargs)
        else:
            yield SampleInput(t, output_shape, **kwargs)