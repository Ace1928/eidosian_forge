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
@overload(operator.eq)
def impl_equal(da, db):
    if not isinstance(da, types.DictType):
        return
    if not isinstance(db, types.DictType):

        def impl_type_mismatch(da, db):
            return False
        return impl_type_mismatch
    otherkeyty = db.key_type

    def impl_type_matched(da, db):
        if len(da) != len(db):
            return False
        for ka, va in da.items():
            kb = _cast(ka, otherkeyty)
            ix, vb = _dict_lookup(db, kb, hash(kb))
            if ix <= DKIX.EMPTY:
                return False
            if va != vb:
                return False
        return True
    return impl_type_matched