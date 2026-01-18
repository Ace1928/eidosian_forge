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
def guard_index(self, idx, msg):
    """
        Raise an error if the index is out of bounds.
        """
    with self._builder.if_then(self.is_out_of_bounds(idx), likely=False):
        self._context.call_conv.return_user_exc(self._builder, IndexError, (msg,))