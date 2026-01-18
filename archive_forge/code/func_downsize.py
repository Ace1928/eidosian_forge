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
def downsize(self, nitems):
    """
        When removing from the set, ensure it is properly sized for the given
        number of used entries.
        """
    context = self._context
    builder = self._builder
    intp_t = nitems.type
    one = ir.Constant(intp_t, 1)
    two = ir.Constant(intp_t, 2)
    minsize = ir.Constant(intp_t, MINSIZE)
    payload = self.payload
    min_entries = builder.shl(nitems, one)
    min_entries = builder.select(builder.icmp_unsigned('>=', min_entries, minsize), min_entries, minsize)
    max_size = builder.shl(min_entries, two)
    size = builder.add(payload.mask, one)
    need_resize = builder.and_(builder.icmp_unsigned('<=', max_size, size), builder.icmp_unsigned('<', minsize, size))
    with builder.if_then(need_resize, likely=False):
        new_size_p = cgutils.alloca_once_value(builder, size)
        bb_body = builder.append_basic_block('calcsize.body')
        bb_end = builder.append_basic_block('calcsize.end')
        builder.branch(bb_body)
        with builder.goto_block(bb_body):
            new_size = builder.load(new_size_p)
            new_size = builder.lshr(new_size, one)
            is_too_small = builder.icmp_unsigned('>', min_entries, new_size)
            with builder.if_then(is_too_small):
                builder.branch(bb_end)
            builder.store(new_size, new_size_p)
            builder.branch(bb_body)
        builder.position_at_end(bb_end)
        new_size = builder.load(new_size_p)
        if DEBUG_ALLOCS:
            context.printf(builder, 'downsize to %zd items: current size = %zd, min entries = %zd, new size = %zd\n', nitems, size, min_entries, new_size)
        self._resize(payload, new_size, 'cannot shrink set')