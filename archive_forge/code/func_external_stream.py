import contextlib
import os
import numpy as np
from .cudadrv import devicearray, devices, driver
from numba.core import config
from numba.cuda.api_util import prepare_shape_strides_dtype
@require_context
def external_stream(ptr):
    """Create a Numba stream object for a stream allocated outside Numba.

    :param ptr: Pointer to the external stream to wrap in a Numba Stream
    :type ptr: int
    """
    return current_context().create_external_stream(ptr)