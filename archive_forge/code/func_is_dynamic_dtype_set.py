import collections
import warnings
from functools import partial, wraps
from typing import Sequence
import numpy as np
import torch
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_dtype import (
from torch.testing._internal.common_utils import torch_to_numpy_dtype_dict
def is_dynamic_dtype_set(op):
    return op.dynamic_dtypes