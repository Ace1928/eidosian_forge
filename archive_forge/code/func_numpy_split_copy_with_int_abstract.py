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
@custom_ops.impl_abstract('_torch_testing::numpy_split_copy_with_int')
def numpy_split_copy_with_int_abstract(x, splits, dim):
    return ([xi.clone() for xi in torch.tensor_split(x, splits, dim)], len(splits))