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
@lower_builtin(operator.iadd, types.List, types.List)
def list_add_inplace(context, builder, sig, args):
    assert sig.args[0].dtype == sig.return_type.dtype
    dest = _list_extend_list(context, builder, sig, args)
    return impl_ret_borrowed(context, builder, sig.return_type, dest.value)