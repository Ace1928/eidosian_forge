import collections
import contextlib
import inspect
import functools
from enum import Enum
from numba.core import typing, types, utils, cgutils
from numba.core.typing.templates import BaseRegistryLoader
@iternext_impl(RefType.BORROWED)
def iternext_wrapper(context, builder, sig, args, result):
    value, = args
    iterobj = cls(context, builder, value)
    return iternext(iterobj, context, builder, result)