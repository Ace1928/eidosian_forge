import math
import llvmlite.ir
import numpy as np
from numba.core.extending import overload
from numba.core.imputils import impl_ret_untracked
from numba.core import typing, types, errors, lowering, cgutils
from numba.core.extending import register_jitable
from numba.np import npdatetime
from numba.cpython import cmathimpl, mathimpl, numbers
def np_int_sdivrem_impl(context, builder, sig, args):
    div = np_int_sdiv_impl(context, builder, sig.return_type[0](*sig.args), args)
    rem = np_int_srem_impl(context, builder, sig.return_type[1](*sig.args), args)
    return context.make_tuple(builder, sig.return_type, [div, rem])