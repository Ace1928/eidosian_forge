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
def check_entry(i):
    """
            Check entry *i* against the value being searched for.
            """
    entry = self.get_entry(i)
    entry_hash = entry.hash
    with builder.if_then(builder.icmp_unsigned('==', h, entry_hash)):
        eq = eqfn(builder, (item, entry.key))
        with builder.if_then(eq):
            builder.branch(bb_found)
    with builder.if_then(is_hash_empty(context, builder, entry_hash)):
        builder.branch(bb_not_found)
    if for_insert:
        with builder.if_then(is_hash_deleted(context, builder, entry_hash)):
            j = builder.load(free_index)
            j = builder.select(builder.icmp_unsigned('==', j, free_index_sentinel), i, j)
            builder.store(j, free_index)