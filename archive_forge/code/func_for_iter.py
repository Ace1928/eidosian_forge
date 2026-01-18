import collections
import contextlib
import inspect
import functools
from enum import Enum
from numba.core import typing, types, utils, cgutils
from numba.core.typing.templates import BaseRegistryLoader
@contextlib.contextmanager
def for_iter(context, builder, iterable_type, val):
    """
    Simulate a for loop on the given iterable.  Yields a namedtuple with
    the given members:
    - `value` is the value being yielded
    - `do_break` is a callable to early out of the loop
    """
    iterator_type = iterable_type.iterator_type
    iterval = call_getiter(context, builder, iterable_type, val)
    bb_body = builder.append_basic_block('for_iter.body')
    bb_end = builder.append_basic_block('for_iter.end')

    def do_break():
        builder.branch(bb_end)
    builder.branch(bb_body)
    with builder.goto_block(bb_body):
        res = call_iternext(context, builder, iterator_type, iterval)
        with builder.if_then(builder.not_(res.is_valid()), likely=False):
            builder.branch(bb_end)
        yield _ForIterLoop(res.yielded_value(), do_break)
        builder.branch(bb_body)
    builder.position_at_end(bb_end)
    if context.enable_nrt:
        context.nrt.decref(builder, iterator_type, iterval)