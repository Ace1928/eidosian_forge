from array import array as native_array
import ctypes
import warnings
import operator
from functools import reduce # pylint: disable=redefined-builtin
import numpy as np
from ..base import _LIB, numeric_types, integer_types
from ..base import c_str, c_array, c_array_buf, c_handle_array, mx_real_t
from ..base import mx_uint, NDArrayHandle, check_call, DLPackHandle, mx_int, mx_int64
from ..base import ctypes2buffer
from ..runtime import Features
from ..context import Context, current_context
from ..util import is_np_array
from . import _internal
from . import op
from ._internal import NDArrayBase
def get_indexing_dispatch_code(key):
    """Returns a dispatch code for calling basic or advanced indexing functions."""
    assert isinstance(key, tuple)
    for idx in key:
        if isinstance(idx, (NDArray, np.ndarray, list, tuple, range)):
            if isinstance(idx, tuple) and len(idx) == 0:
                return _NDARRAY_EMPTY_TUPLE_INDEXING
            return _NDARRAY_ADVANCED_INDEXING
        elif not (isinstance(idx, (py_slice, integer_types)) or idx is None):
            raise ValueError('NDArray does not support slicing with key {} of type {}.'.format(idx, type(idx)))
    return _NDARRAY_BASIC_INDEXING