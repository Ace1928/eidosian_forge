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
@overload(np.nan_to_num)
def nan_to_num_impl(x, copy=True, nan=0.0):
    if isinstance(x, types.Number):
        if isinstance(x, types.Integer):

            def impl(x, copy=True, nan=0.0):
                return x
        elif isinstance(x, types.Float):

            def impl(x, copy=True, nan=0.0):
                if np.isnan(x):
                    return nan
                elif np.isneginf(x):
                    return np.finfo(type(x)).min
                elif np.isposinf(x):
                    return np.finfo(type(x)).max
                return x
        elif isinstance(x, types.Complex):

            def impl(x, copy=True, nan=0.0):
                r = np.nan_to_num(x.real, nan=nan)
                c = np.nan_to_num(x.imag, nan=nan)
                return complex(r, c)
        else:
            raise errors.TypingError('Only Integer, Float, and Complex values are accepted')
    elif type_can_asarray(x):
        if isinstance(x.dtype, types.Integer):

            def impl(x, copy=True, nan=0.0):
                return x
        elif isinstance(x.dtype, types.Float):

            def impl(x, copy=True, nan=0.0):
                min_inf = np.finfo(x.dtype).min
                max_inf = np.finfo(x.dtype).max
                x_ = np.asarray(x)
                output = np.copy(x_) if copy else x_
                output_flat = output.flat
                for i in range(output.size):
                    if np.isnan(output_flat[i]):
                        output_flat[i] = nan
                    elif np.isneginf(output_flat[i]):
                        output_flat[i] = min_inf
                    elif np.isposinf(output_flat[i]):
                        output_flat[i] = max_inf
                return output
        elif isinstance(x.dtype, types.Complex):

            def impl(x, copy=True, nan=0.0):
                x_ = np.asarray(x)
                output = np.copy(x_) if copy else x_
                np.nan_to_num(output.real, copy=False, nan=nan)
                np.nan_to_num(output.imag, copy=False, nan=nan)
                return output
        else:
            raise errors.TypingError('Only Integer, Float, and Complex values are accepted')
    else:
        raise errors.TypingError('The first argument must be a scalar or an array-like')
    return impl