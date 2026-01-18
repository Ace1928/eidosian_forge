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
def sample_inputs_diagonal_scatter(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)
    shapes_2d = ((M, M), (3, 5), (5, 3))
    shapes_3d = ((M, M, M),)
    args_2d = ((), (2,), (-2,), (1,))
    args_3d = ((1, 1, 2), (2, 0, 1), (-2, 0, 1))
    for input_shape, arg in chain(product(shapes_2d, args_2d), product(shapes_3d, args_3d)):
        input_ = make_arg(input_shape)
        if not isinstance(arg, tuple):
            arg_tuple = (arg,)
        else:
            arg_tuple = arg
        src_shape = input_.diagonal(*arg_tuple).size()
        src = make_arg(src_shape)
        yield SampleInput(input_, args=(src, *arg_tuple))