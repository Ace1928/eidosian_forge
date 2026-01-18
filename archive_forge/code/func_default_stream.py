import contextlib
import os
import numpy as np
from .cudadrv import devicearray, devices, driver
from numba.core import config
from numba.cuda.api_util import prepare_shape_strides_dtype
@require_context
def default_stream():
    """
    Get the default CUDA stream. CUDA semantics in general are that the default
    stream is either the legacy default stream or the per-thread default stream
    depending on which CUDA APIs are in use. In Numba, the APIs for the legacy
    default stream are always the ones in use, but an option to use APIs for
    the per-thread default stream may be provided in future.
    """
    return current_context().get_default_stream()