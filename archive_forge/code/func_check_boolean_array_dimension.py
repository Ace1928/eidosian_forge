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
def check_boolean_array_dimension(array_shape, axis, bool_shape):
    """
    Advanced boolean indexing is implemented through the use of `nonzero`.
    Size check is necessary to make sure that the boolean array
    has exactly as many dimensions as it is supposed to work with before the conversion
    """
    for i, val in enumerate(bool_shape):
        if array_shape[axis + i] != val:
            raise IndexError('boolean index did not match indexed array along axis {}; size is {} but corresponding boolean size is {}'.format(axis + i, array_shape[axis + i], val))