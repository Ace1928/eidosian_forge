import operator
from enum import IntEnum
from llvmlite import ir
from numba.core.extending import (
from numba.core.imputils import iternext_impl
from numba.core import types, cgutils
from numba.core.types import (
from numba.core.imputils import impl_ret_borrowed, RefType
from numba.core.errors import TypingError
from numba.core import typing
from numba.typed.typedobjectutils import (_as_bytes, _cast, _nonoptional,
from numba.cpython import listobj
def impl_not_none(this, other):

    def equals(this, other):
        if len(this) != len(other):
            return False
        for i in range(len(this)):
            if this[i] != other[i]:
                return False
        else:
            return True
    return OP(equals(this, other))