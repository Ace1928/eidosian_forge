import collections
import warnings
from functools import partial, wraps
from typing import Sequence
import numpy as np
import torch
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_dtype import (
from torch.testing._internal.common_utils import torch_to_numpy_dtype_dict
def np_unary_ufunc_integer_promotion_wrapper(fn):

    def is_integral(dtype):
        return dtype in [np.bool_, bool, np.uint8, np.int8, np.int16, np.int32, np.int64]

    @wraps(fn)
    def wrapped_fn(x):
        np_dtype = torch_to_numpy_dtype_dict[torch.get_default_dtype()]
        if is_integral(x.dtype):
            return fn(x.astype(np_dtype))
        return fn(x)
    return wrapped_fn