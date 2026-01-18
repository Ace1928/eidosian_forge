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
def array_complex_attr(context, builder, typ, value, attr):
    """
    Given a complex array, it's memory layout is:

        R C R C R C
        ^   ^   ^

    (`R` indicates a float for the real part;
     `C` indicates a float for the imaginary part;
     the `^` indicates the start of each element)

    To get the real part, we can simply change the dtype and itemsize to that
    of the underlying float type.  The new layout is:

        R x R x R x
        ^   ^   ^

    (`x` indicates unused)

    A load operation will use the dtype to determine the number of bytes to
    load.

    To get the imaginary part, we shift the pointer by 1 float offset and
    change the dtype and itemsize.  The new layout is:

        x C x C x C
          ^   ^   ^
    """
    if attr not in ['real', 'imag'] or typ.dtype not in types.complex_domain:
        raise NotImplementedError('cannot get attribute `{}`'.format(attr))
    arrayty = make_array(typ)
    array = arrayty(context, builder, value)
    flty = typ.dtype.underlying_float
    sizeof_flty = context.get_abi_sizeof(context.get_data_type(flty))
    itemsize = array.itemsize.type(sizeof_flty)
    llfltptrty = context.get_value_type(flty).as_pointer()
    dataptr = builder.bitcast(array.data, llfltptrty)
    if attr == 'imag':
        dataptr = builder.gep(dataptr, [ir.IntType(32)(1)])
    resultty = typ.copy(dtype=flty, layout='A')
    result = make_array(resultty)(context, builder)
    repl = dict(data=dataptr, itemsize=itemsize)
    cgutils.copy_struct(result, array, repl)
    return impl_ret_borrowed(context, builder, resultty, result._getvalue())