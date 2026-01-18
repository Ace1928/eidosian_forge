import math
import numbers
import numpy as np
import operator
from llvmlite import ir
from llvmlite.ir import Constant
from numba.core.imputils import (lower_builtin, lower_getattr,
from numba.core import typing, types, utils, errors, cgutils, optional
from numba.core.extending import intrinsic, overload_method
from numba.cpython.unsafe.numbers import viewer
def complex_add_impl(context, builder, sig, args):
    [cx, cy] = args
    ty = sig.args[0]
    x = context.make_complex(builder, ty, value=cx)
    y = context.make_complex(builder, ty, value=cy)
    z = context.make_complex(builder, ty)
    a = x.real
    b = x.imag
    c = y.real
    d = y.imag
    z.real = builder.fadd(a, c)
    z.imag = builder.fadd(b, d)
    res = z._getvalue()
    return impl_ret_untracked(context, builder, sig.return_type, res)