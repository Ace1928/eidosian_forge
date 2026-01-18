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
def clamp_index(self, idx):
    """
        Clamp the index in [0, size].
        """
    builder = self._builder
    idxptr = cgutils.alloca_once_value(builder, idx)
    zero = ir.Constant(idx.type, 0)
    size = self.size
    underflow = self._builder.icmp_signed('<', idx, zero)
    with builder.if_then(underflow, likely=False):
        builder.store(zero, idxptr)
    overflow = self._builder.icmp_signed('>=', idx, size)
    with builder.if_then(overflow, likely=False):
        builder.store(size, idxptr)
    return builder.load(idxptr)