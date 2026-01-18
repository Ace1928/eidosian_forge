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
@register_jitable
def compare_not_none(this, other):
    """Oldschool (python 2.x) cmp.

       if this < other return -1
       if this = other return 0
       if this > other return 1
    """
    if len(this) != len(other):
        return -1 if len(this) < len(other) else 1
    for i in range(len(this)):
        this_item, other_item = (this[i], other[i])
        if this_item != other_item:
            return -1 if this_item < other_item else 1
    else:
        return 0