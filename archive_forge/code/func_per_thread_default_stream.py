import contextlib
import os
import numpy as np
from .cudadrv import devicearray, devices, driver
from numba.core import config
from numba.cuda.api_util import prepare_shape_strides_dtype
@require_context
def per_thread_default_stream():
    """
    Get the per-thread default CUDA stream.
    """
    return current_context().get_per_thread_default_stream()