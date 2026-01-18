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
def impl_integer(l, index, item):
    index = handle_index(l, index)
    castedindex = _cast(index, indexty)
    casteditem = _cast(item, itemty)
    status = _list_setitem(l, castedindex, casteditem)
    if status == ListStatus.LIST_OK:
        return
    elif status == ListStatus.LIST_ERR_IMMUTABLE:
        raise ValueError('list is immutable')
    else:
        raise AssertionError('internal list error during settitem')