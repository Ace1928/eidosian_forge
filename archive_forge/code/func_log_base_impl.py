import cmath
import math
from numba.core.imputils import Registry, impl_ret_untracked
from numba.core import types, cgutils
from numba.core.typing import signature
from numba.cpython import builtins, mathimpl
from numba.core.extending import overload
@lower(cmath.log, types.Complex, types.Complex)
def log_base_impl(context, builder, sig, args):
    """cmath.log(z, base)"""
    [z, base] = args

    def log_base(z, base):
        return cmath.log(z) / cmath.log(base)
    res = context.compile_internal(builder, log_base, sig, args)
    return impl_ret_untracked(context, builder, sig, res)