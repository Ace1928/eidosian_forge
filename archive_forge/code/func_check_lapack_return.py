import contextlib
import warnings
from llvmlite import ir
import numpy as np
import operator
from numba.core.imputils import (lower_builtin, impl_ret_borrowed,
from numba.core.typing import signature
from numba.core.extending import intrinsic, overload, register_jitable
from numba.core import types, cgutils
from numba.core.errors import TypingError, NumbaTypeError, \
from .arrayobj import make_array, _empty_nd_impl, array_copy
from numba.np import numpy_support as np_support
def check_lapack_return(context, builder, res):
    """
    Check the integer error return from one of the LAPACK wrappers in
    _helperlib.c.
    """
    with builder.if_then(cgutils.is_not_null(builder, res), likely=False):
        pyapi = context.get_python_api(builder)
        pyapi.gil_ensure()
        pyapi.fatal_error('LAPACK wrapper returned with an error')