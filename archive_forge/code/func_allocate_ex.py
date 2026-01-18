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
def allocate_ex(cls, context, builder, set_type, nitems=None):
    """
        Allocate a SetInstance with its storage.
        Return a (ok, instance) tuple where *ok* is a LLVM boolean and
        *instance* is a SetInstance object (the object's contents are
        only valid when *ok* is true).
        """
    intp_t = context.get_value_type(types.intp)
    if nitems is None:
        nentries = ir.Constant(intp_t, MINSIZE)
    else:
        if isinstance(nitems, int):
            nitems = ir.Constant(intp_t, nitems)
        nentries = cls.choose_alloc_size(context, builder, nitems)
    self = cls(context, builder, set_type, None)
    ok = self._allocate_payload(nentries)
    return (ok, self)