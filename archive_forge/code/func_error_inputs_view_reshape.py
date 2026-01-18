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
def error_inputs_view_reshape(op, device, **kwargs):
    cases = (((2,), (), False), ((1, 3, 0), (), False), ((4, 3), (4, 2), True), ((1, 3, 5), (5, 2, 2), True), ((1, 3, 5), (5, -1, 2), False), ((1, 3, 5), (5, -1, -1), False), (1, (0, -1), False), ((0, 5), (0, -1), False))
    make_arg = partial(make_tensor, dtype=torch.float32, device=device, requires_grad=False)
    for a, b, is_tensor_supported in cases:
        if kwargs.get('tensor_arg') and (not is_tensor_supported):
            continue
        if b == (5, -1, -1):
            error_regex = 'only one dimension can be inferred'
        elif a == (0, 5):
            error_regex = 'cannot reshape tensor of 0 elements into shape \\[0, -1\\] because the unspecified dimension size -1 can be any value and is ambiguous'
        else:
            shape = ', '.join(map(str, b))
            size = a if type(a) is int else functools.reduce(operator.mul, a, 1)
            error_regex = f"shape '\\[{shape}\\]' is invalid for input of size {size}"
        if kwargs.get('tensor_arg'):
            b = make_arg(b, requires_grad=False)
        yield ErrorInput(SampleInput(make_arg(a), args=(b,)), error_type=Exception, error_regex=error_regex)