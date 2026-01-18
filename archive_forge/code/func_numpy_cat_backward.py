import torch
import functools
from torch.testing import make_tensor
from torch.testing._internal.opinfo.core import (
from torch.testing._internal.common_dtype import all_types_and
import numpy as np
from torch.testing._internal.autograd_function_db import (
from torch import Tensor
from torch.types import Number
from typing import *  # noqa: F403
import torch._custom_ops as custom_ops
@custom_ops.impl_backward('_torch_testing::numpy_cat')
def numpy_cat_backward(ctx, saved, grad_out):
    dim_sizes, dim = saved
    splits = list(np.cumsum(dim_sizes)[:-1])
    grad_xs = torch.ops._torch_testing.numpy_split_copy(grad_out, splits, dim)
    return {'xs': grad_xs}