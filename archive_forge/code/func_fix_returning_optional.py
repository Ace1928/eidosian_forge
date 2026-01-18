import collections
import contextlib
import inspect
import functools
from enum import Enum
from numba.core import typing, types, utils, cgutils
from numba.core.typing.templates import BaseRegistryLoader
def fix_returning_optional(context, builder, sig, status, retval):
    if isinstance(sig.return_type, types.Optional):
        value_type = sig.return_type.type
        optional_none = context.make_optional_none(builder, value_type)
        retvalptr = cgutils.alloca_once_value(builder, optional_none)
        with builder.if_then(builder.not_(status.is_none)):
            optional_value = context.make_optional_value(builder, value_type, retval)
            builder.store(optional_value, retvalptr)
        retval = builder.load(retvalptr)
    return retval