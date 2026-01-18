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
@overload_method(types.ListType, 'remove')
def impl_remove(l, item):
    if not isinstance(l, types.ListType):
        return
    _check_for_none_typed(l, 'remove')
    itemty = l.item_type

    def impl(l, item):
        casteditem = _cast(item, itemty)
        for i, n in enumerate(l):
            if casteditem == n:
                del l[i]
                return
        else:
            raise ValueError('list.remove(x): x not in list')
    return impl