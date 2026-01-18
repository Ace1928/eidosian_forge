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
@overload_method(types.DictType, '__setitem__')
@overload(operator.setitem)
def impl_setitem(d, key, value):
    if not isinstance(d, types.DictType):
        return
    keyty, valty = (d.key_type, d.value_type)

    def impl(d, key, value):
        castedkey = _cast(key, keyty)
        castedval = _cast(value, valty)
        status = _dict_insert(d, castedkey, hash(castedkey), castedval)
        if status == Status.OK:
            return
        elif status == Status.OK_REPLACED:
            return
        elif status == Status.ERR_CMP_FAILED:
            raise ValueError('key comparison failed')
        else:
            raise RuntimeError('dict.__setitem__ failed unexpectedly')
    if d.is_precise():
        return impl
    else:
        d = d.refine(key, value)
        keyty, valty = (d.key_type, d.value_type)
        sig = typing.signature(types.void, d, keyty, valty)
        return (sig, impl)