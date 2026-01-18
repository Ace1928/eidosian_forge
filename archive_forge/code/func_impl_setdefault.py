import ctypes
import operator
from enum import IntEnum
from llvmlite import ir
from numba import _helperlib
from numba.core.extending import (
from numba.core.imputils import iternext_impl, impl_ret_untracked
from numba.core import types, cgutils
from numba.core.types import (
from numba.core.imputils import impl_ret_borrowed, RefType
from numba.core.errors import TypingError, LoweringError
from numba.core import typing
from numba.typed.typedobjectutils import (_as_bytes, _cast, _nonoptional,
@overload_method(types.DictType, 'setdefault')
def impl_setdefault(dct, key, default=None):
    if not isinstance(dct, types.DictType):
        return

    def impl(dct, key, default=None):
        if key not in dct:
            dct[key] = default
        return dct[key]
    return impl