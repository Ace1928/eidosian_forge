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
@contextlib.contextmanager
def make_contiguous(context, builder, sig, args):
    """
    Ensure that all array arguments are contiguous, if necessary by
    copying them.
    A new (sig, args) tuple is yielded.
    """
    newtys = []
    newargs = []
    copies = []
    for ty, val in zip(sig.args, args):
        if not isinstance(ty, types.Array) or ty.layout in 'CF':
            newty, newval = (ty, val)
        else:
            newty = ty.copy(layout='C')
            copysig = signature(newty, ty)
            newval = array_copy(context, builder, copysig, (val,))
            copies.append((newty, newval))
        newtys.append(newty)
        newargs.append(newval)
    yield (signature(sig.return_type, *newtys), tuple(newargs))
    for ty, val in copies:
        context.nrt.decref(builder, ty, val)