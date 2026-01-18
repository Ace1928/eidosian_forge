import collections
import contextlib
import inspect
import functools
from enum import Enum
from numba.core import typing, types, utils, cgutils
from numba.core.typing.templates import BaseRegistryLoader
def call_len(context, builder, ty, val):
    """
    Call len() on the given value.  Return None if len() isn't defined on
    this type.
    """
    try:
        len_impl = context.get_function(len, typing.signature(types.intp, ty))
    except NotImplementedError:
        return None
    else:
        return len_impl(builder, (val,))