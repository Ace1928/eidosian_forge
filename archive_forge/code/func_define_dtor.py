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
def define_dtor(self):
    """Define the destructor if not already defined"""
    context = self._context
    builder = self._builder
    mod = builder.module
    fnty = ir.FunctionType(ir.VoidType(), [cgutils.voidptr_t])
    fn = cgutils.get_or_insert_function(mod, fnty, '.dtor.list.{}'.format(self.dtype))
    if not fn.is_declaration:
        return fn
    fn.linkage = 'linkonce_odr'
    builder = ir.IRBuilder(fn.append_basic_block())
    base_ptr = fn.args[0]
    payload = ListPayloadAccessor(context, builder, self._ty, base_ptr)
    intp = payload.size.type
    with cgutils.for_range_slice(builder, start=intp(0), stop=payload.size, step=intp(1), intp=intp) as (idx, _):
        val = payload.getitem(idx)
        context.nrt.decref(builder, self.dtype, val)
    builder.ret_void()
    return fn