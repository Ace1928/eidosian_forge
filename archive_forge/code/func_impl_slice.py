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
def impl_slice(l, index, item):
    if not l._is_mutable():
        raise ValueError('list is immutable')
    if l is item:
        item = item.copy()
    slice_range = handle_slice(l, index)
    if slice_range.step == 1:
        if len(item) == len(slice_range):
            for i, j in zip(slice_range, item):
                l[i] = j
        if len(item) > len(slice_range):
            for i, j in zip(slice_range, item[:len(slice_range)]):
                l[i] = j
            insert_range = range(slice_range.stop, slice_range.stop + len(item) - len(slice_range))
            for i, k in zip(insert_range, item[len(slice_range):]):
                l.insert(i, k)
        if len(item) < len(slice_range):
            replace_range = range(slice_range.start, slice_range.start + len(item))
            for i, j in zip(replace_range, item):
                l[i] = j
            del l[slice_range.start + len(item):slice_range.stop]
    else:
        if len(slice_range) != len(item):
            raise ValueError('length mismatch for extended slice and sequence')
        for i, j in zip(slice_range, item):
            l[i] = j