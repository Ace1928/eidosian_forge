import math
import llvmlite.ir
import numpy as np
from numba.core.extending import overload
from numba.core.imputils import impl_ret_untracked
from numba.core import typing, types, errors, lowering, cgutils
from numba.core.extending import register_jitable
from numba.np import npdatetime
from numba.cpython import cmathimpl, mathimpl, numbers
def np_complex_square_impl(context, builder, sig, args):
    _check_arity_and_homogeneity(sig, args, 1)
    binary_sig = typing.signature(*[sig.return_type] * 3)
    return numbers.complex_mul_impl(context, builder, binary_sig, [args[0], args[0]])