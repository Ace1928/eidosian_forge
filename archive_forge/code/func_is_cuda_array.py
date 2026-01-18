import contextlib
import os
import numpy as np
from .cudadrv import devicearray, devices, driver
from numba.core import config
from numba.cuda.api_util import prepare_shape_strides_dtype
def is_cuda_array(obj):
    """Test if the object has defined the `__cuda_array_interface__` attribute.

    Does not verify the validity of the interface.
    """
    return hasattr(obj, '__cuda_array_interface__')