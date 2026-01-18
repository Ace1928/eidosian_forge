import operator
from numba.core import types, typing, cgutils
from numba.core.imputils import (lower_cast, lower_builtin,
def always_return_false_impl(context, builder, sig, args):
    return cgutils.false_bit