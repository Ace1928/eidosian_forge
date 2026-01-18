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
def complex_negate_impl(context, builder, sig, args):
    from numba.cpython import mathimpl
    [typ] = sig.args
    [val] = args
    cmplx = context.make_complex(builder, typ, value=val)
    res = context.make_complex(builder, typ)
    res.real = mathimpl.negate_real(builder, cmplx.real)
    res.imag = mathimpl.negate_real(builder, cmplx.imag)
    res = res._getvalue()
    return impl_ret_untracked(context, builder, sig.return_type, res)