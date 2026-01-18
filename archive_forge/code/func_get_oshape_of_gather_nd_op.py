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
def get_oshape_of_gather_nd_op(dshape, ishape):
    """Given data and index shapes, get the output `NDArray` shape.
    This basically implements the infer shape logic of op gather_nd."""
    assert len(dshape) > 0 and len(ishape) > 0
    oshape = list(ishape[1:])
    if ishape[0] < len(dshape):
        oshape.extend(dshape[ishape[0]:])
    return tuple(oshape)