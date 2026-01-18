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
@overload_method(types.ListType, 'append')
def impl_append(l, item):
    if not isinstance(l, types.ListType):
        return
    itemty = l.item_type

    def impl(l, item):
        casteditem = _cast(item, itemty)
        status = _list_append(l, casteditem)
        if status == ListStatus.LIST_OK:
            return
        elif status == ListStatus.LIST_ERR_IMMUTABLE:
            raise ValueError('list is immutable')
        elif status == ListStatus.LIST_ERR_NO_MEMORY:
            raise MemoryError('Unable to allocate memory to append item')
        else:
            raise RuntimeError('list.append failed unexpectedly')
    if l.is_precise():
        return impl
    else:
        l = l.refine(item)
        itemty = l.item_type
        sig = typing.signature(types.void, l, itemty)
        return (sig, impl)