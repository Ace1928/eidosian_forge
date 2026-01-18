import collections
import contextlib
import inspect
import functools
from enum import Enum
from numba.core import typing, types, utils, cgutils
from numba.core.typing.templates import BaseRegistryLoader
def call_iternext(context, builder, iterator_type, val):
    """
    Call the `iternext()` implementation for the given *iterator_type*
    of value *val*, and return a convenience _IternextResult() object
    reflecting the results.
    """
    itemty = iterator_type.yield_type
    pair_type = types.Pair(itemty, types.boolean)
    iternext_sig = typing.signature(pair_type, iterator_type)
    iternext_impl = context.get_function('iternext', iternext_sig)
    val = iternext_impl(builder, (val,))
    pairobj = context.make_helper(builder, pair_type, val)
    return _IternextResult(context, builder, pairobj)