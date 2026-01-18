import collections
import contextlib
import inspect
import functools
from enum import Enum
from numba.core import typing, types, utils, cgutils
from numba.core.typing.templates import BaseRegistryLoader
def impl_ret_borrowed(ctx, builder, retty, ret):
    """
    The implementation returns a borrowed reference.
    This function automatically incref so that the implementation is
    returning a new reference.
    """
    if ctx.enable_nrt:
        ctx.nrt.incref(builder, retty, ret)
    return ret