import contextlib
import os
import numpy as np
from .cudadrv import devicearray, devices, driver
from numba.core import config
from numba.cuda.api_util import prepare_shape_strides_dtype
def as_cuda_array(obj, sync=True):
    """Create a DeviceNDArray from any object that implements
    the :ref:`cuda array interface <cuda-array-interface>`.

    A view of the underlying GPU buffer is created.  No copying of the data
    is done.  The resulting DeviceNDArray will acquire a reference from `obj`.

    If ``sync`` is ``True``, then the imported stream (if present) will be
    synchronized.
    """
    if not is_cuda_array(obj):
        raise TypeError("*obj* doesn't implement the cuda array interface.")
    else:
        return from_cuda_array_interface(obj.__cuda_array_interface__, owner=obj, sync=sync)