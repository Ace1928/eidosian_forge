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
def check_seq_size(seqty, seq, shapes):
    if len(shapes) == 0:
        return
    size = _get_seq_size(context, builder, seqty, seq)
    expected = shapes[0]
    mismatch = builder.icmp_signed('!=', size, expected)
    with builder.if_then(mismatch, likely=False):
        _fail()
    if len(shapes) == 1:
        return
    if isinstance(seqty, types.Sequence):
        getitem_impl = _get_borrowing_getitem(context, seqty)
        with cgutils.for_range(builder, size) as loop:
            innerty = seqty.dtype
            inner = getitem_impl(builder, (seq, loop.index))
            check_seq_size(innerty, inner, shapes[1:])
    elif isinstance(seqty, types.BaseTuple):
        for i in range(len(seqty)):
            innerty = seqty[i]
            inner = builder.extract_value(seq, i)
            check_seq_size(innerty, inner, shapes[1:])
    else:
        assert 0, seqty