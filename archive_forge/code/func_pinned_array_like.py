from contextlib import contextmanager
import numpy as np
from_record_like = None
def pinned_array_like(ary):
    strides = _contiguous_strides_like_array(ary)
    order = _order_like_array(ary)
    return pinned_array(shape=ary.shape, dtype=ary.dtype, strides=strides, order=order)