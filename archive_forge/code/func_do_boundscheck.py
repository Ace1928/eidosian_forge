import collections
from contextlib import contextmanager, ExitStack
import functools
from llvmlite import ir
from numba.core import utils, types, config, debuginfo
import numba.core.datamodel
def do_boundscheck(context, builder, ind, dimlen, axis=None):

    def _dbg():
        if axis is not None:
            if isinstance(axis, int):
                printf(builder, 'debug: IndexError: index %d is out of bounds for axis {} with size %d\n'.format(axis), ind, dimlen)
            else:
                printf(builder, 'debug: IndexError: index %d is out of bounds for axis %d with size %d\n', ind, axis, dimlen)
        else:
            printf(builder, 'debug: IndexError: index %d is out of bounds for size %d\n', ind, dimlen)
    msg = 'index is out of bounds'
    out_of_bounds_upper = builder.icmp_signed('>=', ind, dimlen)
    with if_unlikely(builder, out_of_bounds_upper):
        if config.FULL_TRACEBACKS:
            _dbg()
        context.call_conv.return_user_exc(builder, IndexError, (msg,))
    out_of_bounds_lower = builder.icmp_signed('<', ind, ind.type(0))
    with if_unlikely(builder, out_of_bounds_lower):
        if config.FULL_TRACEBACKS:
            _dbg()
        context.call_conv.return_user_exc(builder, IndexError, (msg,))