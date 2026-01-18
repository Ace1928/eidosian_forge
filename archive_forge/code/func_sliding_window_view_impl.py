import functools
import math
import operator
from llvmlite import ir
from llvmlite.ir import Constant
import numpy as np
from numba import pndindex, literal_unroll
from numba.core import types, typing, errors, cgutils, extending
from numba.np.numpy_support import (as_dtype, from_dtype, carray, farray,
from numba.np.numpy_support import type_can_asarray, is_nonelike, numpy_version
from numba.core.imputils import (lower_builtin, lower_getattr,
from numba.core.typing import signature
from numba.core.types import StringLiteral
from numba.core.extending import (register_jitable, overload, overload_method,
from numba.misc import quicksort, mergesort
from numba.cpython import slicing
from numba.cpython.unsafe.tuple import tuple_setitem, build_full_slice_tuple
from numba.core.extending import overload_classmethod
from numba.core.typing.npydecl import (parse_dtype as ty_parse_dtype,
def sliding_window_view_impl(x, window_shape, axis=None):
    window_shape = get_window_shape(window_shape)
    axis = get_axis(window_shape, axis, x.ndim)
    if len(window_shape) != len(axis):
        raise ValueError('Must provide matching length window_shape and axis')
    out_shape = shape_buffer
    out_strides = stride_buffer
    for i in range(x.ndim):
        out_shape = tuple_setitem(out_shape, i, x.shape[i])
        out_strides = tuple_setitem(out_strides, i, x.strides[i])
    i = x.ndim
    for ax, dim in zip(axis, window_shape):
        if dim < 0:
            raise ValueError('`window_shape` cannot contain negative values')
        if out_shape[ax] < dim:
            raise ValueError('window_shape cannot be larger than input array shape')
        trimmed = out_shape[ax] - dim + 1
        out_shape = tuple_setitem(out_shape, ax, trimmed)
        out_shape = tuple_setitem(out_shape, i, dim)
        out_strides = tuple_setitem(out_strides, i, x.strides[ax])
        i += 1
    view = reshape_unchecked(x, out_shape, out_strides)
    return view