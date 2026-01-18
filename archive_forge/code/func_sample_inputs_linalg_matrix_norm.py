import itertools
import random
import unittest
from functools import partial
from itertools import chain, product
from typing import Iterable, List
import numpy as np
from numpy import inf
import torch
from torch.testing import make_tensor
from torch.testing._internal.common_cuda import (
from torch.testing._internal.common_device_type import (
from torch.testing._internal.common_dtype import (
from torch.testing._internal.common_utils import (
from torch.testing._internal.opinfo.core import (
from torch.testing._internal.opinfo.refs import PythonRefInfo, ReductionPythonRefInfo
def sample_inputs_linalg_matrix_norm(op_info, device, dtype, requires_grad, **kwargs):
    low_precision_dtypes = (torch.float16, torch.bfloat16, torch.complex32)
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    sizes = ((2, 2), (2, 3, 2))
    if dtype in low_precision_dtypes:
        ords = ('fro', inf, -inf, 1, -1)
    else:
        ords = ('fro', 'nuc', inf, -inf, 1, -1, 2, -2)
    dims = ((-2, -1), (-1, 0))
    for size, ord, dim, keepdim in product(sizes, ords, dims, [True, False]):
        yield SampleInput(make_arg(size), args=(ord, dim, keepdim))