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
@lower_builtin('list.clear', types.List)
def list_clear(context, builder, sig, args):
    inst = ListInstance(context, builder, sig.args[0], args[0])
    inst.resize(context.get_constant(types.intp, 0))
    return context.get_dummy_value()