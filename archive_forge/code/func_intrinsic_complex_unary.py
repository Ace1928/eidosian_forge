import cmath
import math
from numba.core.imputils import Registry, impl_ret_untracked
from numba.core import types, cgutils
from numba.core.typing import signature
from numba.cpython import builtins, mathimpl
from numba.core.extending import overload
def intrinsic_complex_unary(inner_func):

    def wrapper(context, builder, sig, args):
        [typ] = sig.args
        [value] = args
        z = context.make_complex(builder, typ, value=value)
        x = z.real
        y = z.imag
        x_is_finite = mathimpl.is_finite(builder, x)
        y_is_finite = mathimpl.is_finite(builder, y)
        inner_sig = signature(sig.return_type, *(typ.underlying_float,) * 2 + (types.boolean,) * 2)
        res = context.compile_internal(builder, inner_func, inner_sig, (x, y, x_is_finite, y_is_finite))
        return impl_ret_untracked(context, builder, sig, res)
    return wrapper