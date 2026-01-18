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
@overload_method(types.ListType, 'insert')
def impl_insert(l, index, item):
    if not isinstance(l, types.ListType):
        return
    _check_for_none_typed(l, 'insert')
    if isinstance(item, NoneType):
        raise TypingError('method support for List[None] is limited')
    if index in index_types:

        def impl(l, index, item):
            if index >= len(l) or len(l) == 0:
                l.append(item)
            else:
                if index < 0:
                    index = max(len(l) + index, 0)
                l.append(l[0])
                i = len(l) - 1
                while i > index:
                    l[i] = l[i - 1]
                    i -= 1
                l[index] = item
        if l.is_precise():
            return impl
        else:
            l = l.refine(item)
            itemty = l.item_type
            sig = typing.signature(types.void, l, INDEXTY, itemty)
            return (sig, impl)
    else:
        raise TypingError('list insert indices must be integers')