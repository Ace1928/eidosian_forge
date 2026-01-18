import functools
import math
import operator
from llvmlite import ir
from llvmlite.ir import Constant
import numpy as np
from numba import pndindex, literal_unroll
from numba.core import types, typing, errors, cgutils, extending
from numba.np.numpy_support import (as_dtype, from_dtype, carray, farray,
from numba.np.numpy_support import type_can_asarray, is_nonelike, numpy_version
from numba.core.imputils import (lower_builtin, lower_getattr,
from numba.core.typing import signature
from numba.core.types import StringLiteral
from numba.core.extending import (register_jitable, overload, overload_method,
from numba.misc import quicksort, mergesort
from numba.cpython import slicing
from numba.cpython.unsafe.tuple import tuple_setitem, build_full_slice_tuple
from numba.core.extending import overload_classmethod
from numba.core.typing.npydecl import (parse_dtype as ty_parse_dtype,
def assign_sequence_to_array(context, builder, data, shapes, strides, arrty, seqty, seq):
    """
    Assign a nested sequence contents to an array.  The shape must match
    the sequence's structure.
    """

    def assign_item(indices, valty, val):
        ptr = cgutils.get_item_pointer2(context, builder, data, shapes, strides, arrty.layout, indices, wraparound=False)
        val = context.cast(builder, val, valty, arrty.dtype)
        store_item(context, builder, arrty, val, ptr)

    def assign(seqty, seq, shapes, indices):
        if len(shapes) == 0:
            assert not isinstance(seqty, (types.Sequence, types.BaseTuple))
            assign_item(indices, seqty, seq)
            return
        size = shapes[0]
        if isinstance(seqty, types.Sequence):
            getitem_impl = _get_borrowing_getitem(context, seqty)
            with cgutils.for_range(builder, size) as loop:
                innerty = seqty.dtype
                inner = getitem_impl(builder, (seq, loop.index))
                assign(innerty, inner, shapes[1:], indices + (loop.index,))
        elif isinstance(seqty, types.BaseTuple):
            for i in range(len(seqty)):
                innerty = seqty[i]
                inner = builder.extract_value(seq, i)
                index = context.get_constant(types.intp, i)
                assign(innerty, inner, shapes[1:], indices + (index,))
        else:
            assert 0, seqty
    assign(seqty, seq, shapes, ())