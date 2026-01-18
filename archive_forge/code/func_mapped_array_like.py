import contextlib
import os
import numpy as np
from .cudadrv import devicearray, devices, driver
from numba.core import config
from numba.cuda.api_util import prepare_shape_strides_dtype
def mapped_array_like(ary, stream=0, portable=False, wc=False):
    """
    Call :func:`mapped_array() <numba.cuda.mapped_array>` with the information
    from the array.
    """
    strides = _contiguous_strides_like_array(ary)
    order = _order_like_array(ary)
    return mapped_array(shape=ary.shape, dtype=ary.dtype, strides=strides, order=order, stream=stream, portable=portable, wc=wc)