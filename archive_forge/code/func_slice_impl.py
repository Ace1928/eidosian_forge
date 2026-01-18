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
def slice_impl(l, index):
    slice_range = handle_slice(l, index)
    status = _list_delete_slice(l, slice_range.start, slice_range.stop, slice_range.step)
    if status == ListStatus.LIST_ERR_MUTATED:
        raise ValueError('list is immutable')