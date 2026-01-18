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
@overload_method(types.DictType, 'popitem')
def impl_popitem(d):
    if not isinstance(d, types.DictType):
        return

    def impl(d):
        status, keyval = _dict_popitem(d)
        if status == Status.OK:
            return _nonoptional(keyval)
        elif status == Status.ERR_DICT_EMPTY:
            raise KeyError()
        else:
            raise AssertionError('internal dict error during popitem')
    return impl