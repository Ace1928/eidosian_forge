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
@overload(operator.getitem)
def impl_getitem(d, key):
    if not isinstance(d, types.DictType):
        return
    keyty = d.key_type

    def impl(d, key):
        castedkey = _cast(key, keyty)
        ix, val = _dict_lookup(d, castedkey, hash(castedkey))
        if ix == DKIX.EMPTY:
            raise KeyError()
        elif ix < DKIX.EMPTY:
            raise AssertionError('internal dict error during lookup')
        else:
            return _nonoptional(val)
    return impl