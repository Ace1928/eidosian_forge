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
def get_hash_value(context, builder, typ, value):
    """
    Compute the hash of the given value.
    """
    typingctx = context.typing_context
    fnty = typingctx.resolve_value_type(hash)
    sig = fnty.get_call_type(typingctx, (typ,), {})
    fn = context.get_function(fnty, sig)
    h = fn(builder, (value,))
    is_ok = is_hash_used(context, builder, h)
    fallback = ir.Constant(h.type, FALLBACK)
    return builder.select(is_ok, h, fallback)