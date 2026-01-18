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
@overload_method(types.LiteralList, 'index')
def literal_list_index(lst, x, start=0, end=_index_end):
    if isinstance(lst, types.LiteralList):
        msg = 'list.index is unsupported for literal lists'
        raise errors.TypingError(msg)