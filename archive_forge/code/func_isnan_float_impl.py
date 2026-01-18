import cmath
import math
from numba.core.imputils import Registry, impl_ret_untracked
from numba.core import types, cgutils
from numba.core.typing import signature
from numba.cpython import builtins, mathimpl
from numba.core.extending import overload
@lower(cmath.isnan, types.Complex)
def isnan_float_impl(context, builder, sig, args):
    [typ] = sig.args
    [value] = args
    z = context.make_complex(builder, typ, value=value)
    res = is_nan(builder, z)
    return impl_ret_untracked(context, builder, sig.return_type, res)