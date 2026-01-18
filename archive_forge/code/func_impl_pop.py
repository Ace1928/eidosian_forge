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
@overload_method(types.DictType, 'pop')
def impl_pop(dct, key, default=None):
    if not isinstance(dct, types.DictType):
        return
    keyty = dct.key_type
    valty = dct.value_type
    should_raise = isinstance(default, types.Omitted)
    _sentry_safe_cast_default(default, valty)

    def impl(dct, key, default=None):
        castedkey = _cast(key, keyty)
        hashed = hash(castedkey)
        ix, val = _dict_lookup(dct, castedkey, hashed)
        if ix == DKIX.EMPTY:
            if should_raise:
                raise KeyError()
            else:
                return default
        elif ix < DKIX.EMPTY:
            raise AssertionError('internal dict error during lookup')
        else:
            status = _dict_delitem(dct, hashed, ix)
            if status != Status.OK:
                raise AssertionError('internal dict error during delitem')
            return val
    return impl