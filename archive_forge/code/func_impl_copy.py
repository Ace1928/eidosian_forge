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
@overload_method(types.DictType, 'copy')
def impl_copy(d):
    if not isinstance(d, types.DictType):
        return
    key_type, val_type = (d.key_type, d.value_type)

    def impl(d):
        newd = new_dict(key_type, val_type, n_keys=len(d))
        for k, v in d.items():
            newd[k] = v
        return newd
    return impl