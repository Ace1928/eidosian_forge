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
def sample_inputs_scaled_mm(op_info, device, dtype, requires_grad, **kwargs):
    make_mat_e4m3 = partial(make_tensor, device=device, dtype=torch.float8_e4m3fn, requires_grad=requires_grad)
    make_mat_e5m2 = partial(make_tensor, device=device, dtype=torch.float8_e5m2, requires_grad=requires_grad)
    M, N, K = (15, 32, 16)
    samples = []
    mat1 = make_mat_e4m3((M, K))
    mat2 = make_mat_e4m3((K, N)).t().contiguous().t()
    samples.append(SampleInput(mat1, mat2))
    mat1 = make_mat_e4m3((M, K))
    mat2 = make_mat_e5m2((K, N)).t().contiguous().t()
    samples.append(SampleInput(mat1, mat2))
    mat1 = make_mat_e5m2((M, K))
    mat2 = make_mat_e4m3((K, N)).t().contiguous().t()
    samples.append(SampleInput(mat1, mat2))
    yield from samples