from numba.core import types, cgutils
from numba.core.imputils import (
@lower_builtin('getiter', types.IteratorType)
def iterator_getiter(context, builder, sig, args):
    [it] = args
    return impl_ret_borrowed(context, builder, sig.return_type, it)