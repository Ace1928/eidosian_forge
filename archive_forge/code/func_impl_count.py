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
@overload_method(types.ListType, 'count')
def impl_count(l, item):
    if not isinstance(l, types.ListType):
        return
    _check_for_none_typed(l, 'count')
    itemty = l.item_type

    def impl(l, item):
        casteditem = _cast(item, itemty)
        total = 0
        for i in l:
            if i == casteditem:
                total += 1
        return total
    return impl