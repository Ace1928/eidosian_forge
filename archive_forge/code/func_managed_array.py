from contextlib import contextmanager
import numpy as np
from_record_like = None
def managed_array(shape, dtype=np.float_, strides=None, order='C'):
    return np.ndarray(shape=shape, strides=strides, dtype=dtype, order=order)