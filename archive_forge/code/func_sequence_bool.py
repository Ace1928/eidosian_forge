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
@lower_builtin(bool, types.Sequence)
def sequence_bool(context, builder, sig, args):

    def sequence_bool_impl(seq):
        return len(seq) != 0
    return context.compile_internal(builder, sequence_bool_impl, sig, args)