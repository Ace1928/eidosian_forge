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
def compare_some_none(this, other, this_is_none, other_is_none):
    """Oldschool (python 2.x) cmp for None typed lists.

       if this < other return -1
       if this = other return 0
       if this > other return 1
    """
    if len(this) != len(other):
        return -1 if len(this) < len(other) else 1
    if this_is_none and other_is_none:
        return 0
    return -1 if this_is_none else 1