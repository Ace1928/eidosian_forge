import math
import operator
from functools import cached_property
from llvmlite import ir
from numba.core import types, typing, errors, cgutils
from numba.core.imputils import (lower_builtin, lower_cast,
from numba.core.extending import overload_method, overload
from numba.misc import quicksort
from numba.cpython import slicing
from numba import literal_unroll
@lower_builtin(operator.getitem, types.List, types.SliceType)
def getslice_list(context, builder, sig, args):
    inst = ListInstance(context, builder, sig.args[0], args[0])
    slice = context.make_helper(builder, sig.args[1], args[1])
    slicing.guard_invalid_slice(context, builder, sig.args[1], slice)
    inst.fix_slice(slice)
    result_size = slicing.get_slice_length(builder, slice)
    result = ListInstance.allocate(context, builder, sig.return_type, result_size)
    result.size = result_size
    with cgutils.for_range_slice_generic(builder, slice.start, slice.stop, slice.step) as (pos_range, neg_range):
        with pos_range as (idx, count):
            value = inst.getitem(idx)
            result.inititem(count, value, incref=True)
        with neg_range as (idx, count):
            value = inst.getitem(idx)
            result.inititem(count, value, incref=True)
    return impl_ret_new_ref(context, builder, sig.return_type, result.value)