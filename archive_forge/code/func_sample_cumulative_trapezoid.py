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
def sample_cumulative_trapezoid(op_info, device, dtype, requires_grad, **kwargs):
    y_shape_x_shape_and_kwargs = [((2, 3), (2, 3), {}), ((2, 3), (2, 3), {'dim': 1}), ((6,), (6,), {}), ((6,), None, {}), ((2, 3), (1, 3), {}), ((3, 3), (3, 3), {}), ((3, 3), (3, 3), {'dim': -2}), ((5,), None, {'dx': 2.0}), ((2, 2), None, {'dx': 3.0})]
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad, low=None, high=None)
    for y_shape, x_shape, kwarg in y_shape_x_shape_and_kwargs:
        y_tensor = make_arg(y_shape)
        if x_shape is not None:
            x_tensor = make_arg(x_shape)
            yield SampleInput(y_tensor, x_tensor, **kwarg)
        else:
            yield SampleInput(y_tensor, **kwarg)