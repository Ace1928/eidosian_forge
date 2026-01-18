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
@overload_method(types.ListType, 'reverse')
def impl_reverse(l):
    if not isinstance(l, types.ListType):
        return
    _check_for_none_typed(l, 'reverse')

    def impl(l):
        if not l._is_mutable():
            raise ValueError('list is immutable')
        front = 0
        back = len(l) - 1
        while front < back:
            l[front], l[back] = (l[back], l[front])
            front += 1
            back -= 1
    return impl