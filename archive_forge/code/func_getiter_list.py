import math
import operator
from functools import cached_property
from llvmlite import ir
from numba.core import types, typing, errors, cgutils
from numba.core.imputils import (lower_builtin, lower_cast,
from numba.core.extending import overload_method, overload
from numba.misc import quicksort
from numba.cpython import slicing
from numba import literal_unroll
@lower_builtin('getiter', types.List)
def getiter_list(context, builder, sig, args):
    inst = ListIterInstance.from_list(context, builder, sig.return_type, args[0])
    return impl_ret_borrowed(context, builder, sig.return_type, inst.value)