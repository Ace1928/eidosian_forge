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
def is_out_of_bounds(self, idx):
    """
        Return whether the index is out of bounds.
        """
    underflow = self._builder.icmp_signed('<', idx, ir.Constant(idx.type, 0))
    overflow = self._builder.icmp_signed('>=', idx, self.size)
    return self._builder.or_(underflow, overflow)