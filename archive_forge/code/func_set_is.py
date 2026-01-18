import collections
import contextlib
import math
import operator
from functools import cached_property
from llvmlite import ir
from numba.core import types, typing, cgutils
from numba.core.imputils import (lower_builtin, lower_cast,
from numba.misc import quicksort
from numba.cpython import slicing
from numba.core.errors import NumbaValueError, TypingError
from numba.core.extending import overload, overload_method, intrinsic
@lower_builtin(operator.is_, types.Set, types.Set)
def set_is(context, builder, sig, args):
    a = SetInstance(context, builder, sig.args[0], args[0])
    b = SetInstance(context, builder, sig.args[1], args[1])
    ma = builder.ptrtoint(a.meminfo, cgutils.intp_t)
    mb = builder.ptrtoint(b.meminfo, cgutils.intp_t)
    return builder.icmp_signed('==', ma, mb)