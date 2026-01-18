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
@classmethod
def numba_xxpotrf(cls, dtype):
    sig = types.intc(types.char, types.char, types.intp, types.CPointer(dtype), types.intp)
    return types.ExternalFunction('numba_xxpotrf', sig)