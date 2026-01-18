import builtins
import math
import torch
from keras.src.backend import KerasTensor
from keras.src.backend import config
from keras.src.backend.common import dtypes
from keras.src.backend.common.backend_utils import canonicalize_axis
from keras.src.backend.common.backend_utils import to_tuple_or_list
from keras.src.backend.common.variables import standardize_dtype
from keras.src.backend.torch.core import cast
from keras.src.backend.torch.core import convert_to_tensor
from keras.src.backend.torch.core import get_device
from keras.src.backend.torch.core import is_tensor
from keras.src.backend.torch.core import to_torch_dtype
def can_use_int_matmul(x1, x2):
    if get_device() != 'cuda':
        return False
    x1_dtype = standardize_dtype(x1.dtype)
    x2_dtype = standardize_dtype(x2.dtype)
    if x1_dtype != 'int8' or x2_dtype != 'int8':
        return False
    x1_shape = x1.shape
    x2_shape = x2.shape
    if x1.ndim != 2 or x2.ndim != 2:
        return False
    if x1_shape[0] <= 16 or x1_shape[1] < 16 or x1_shape[1] % 8 != 0:
        return False
    if x2_shape[0] < 16 or x2_shape[0] % 8 != 0 or x2_shape[1] % 8 != 0:
        return False
    return True