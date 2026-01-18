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
@intrinsic
def np_frombuffer(typingctx, buffer, dtype, retty):
    ty = retty.instance_type
    sig = ty(buffer, dtype, retty)

    def codegen(context, builder, sig, args):
        bufty = sig.args[0]
        aryty = sig.return_type
        buf = make_array(bufty)(context, builder, value=args[0])
        out_ary_ty = make_array(aryty)
        out_ary = out_ary_ty(context, builder)
        out_datamodel = out_ary._datamodel
        itemsize = get_itemsize(context, aryty)
        ll_itemsize = Constant(buf.itemsize.type, itemsize)
        nbytes = builder.mul(buf.nitems, buf.itemsize)
        rem = builder.srem(nbytes, ll_itemsize)
        is_incompatible = cgutils.is_not_null(builder, rem)
        with builder.if_then(is_incompatible, likely=False):
            msg = 'buffer size must be a multiple of element size'
            context.call_conv.return_user_exc(builder, ValueError, (msg,))
        shape = cgutils.pack_array(builder, [builder.sdiv(nbytes, ll_itemsize)])
        strides = cgutils.pack_array(builder, [ll_itemsize])
        data = builder.bitcast(buf.data, context.get_value_type(out_datamodel.get_type('data')))
        populate_array(out_ary, data=data, shape=shape, strides=strides, itemsize=ll_itemsize, meminfo=buf.meminfo, parent=buf.parent)
        res = out_ary._getvalue()
        return impl_ret_borrowed(context, builder, sig.return_type, res)
    return (sig, codegen)