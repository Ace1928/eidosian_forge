from functools import singledispatch
from llvmlite import ir
from numba.core import types, cgutils
from numba.core.errors import NumbaWarning
from numba.core.imputils import Registry
from numba.cuda import nvvmutils
from warnings import warn
@singledispatch
def print_item(ty, context, builder, val):
    """
    Handle printing of a single value of the given Numba type.
    A (format string, [list of arguments]) is returned that will allow
    forming the final printf()-like call.
    """
    raise NotImplementedError('printing unimplemented for values of type %s' % (ty,))