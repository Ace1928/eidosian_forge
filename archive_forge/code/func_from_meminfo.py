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
@classmethod
def from_meminfo(cls, context, builder, set_type, meminfo):
    """
        Allocate a new set instance pointing to an existing payload
        (a meminfo pointer).
        Note the parent field has to be filled by the caller.
        """
    self = cls(context, builder, set_type, None)
    self._set.meminfo = meminfo
    self._set.parent = context.get_constant_null(types.pyobject)
    context.nrt.incref(builder, set_type, self.value)
    return self