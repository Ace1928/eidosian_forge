import math
import random
import numpy as np
from llvmlite import ir
from numba.core.cgutils import is_nonelike
from numba.core.extending import intrinsic, overload, register_jitable
from numba.core.imputils import (Registry, impl_ret_untracked,
from numba.core.typing import signature
from numba.core import types, cgutils
from numba.np import arrayobj
from numba.core.errors import NumbaTypeError
@overload(np.random.random)
@overload(np.random.random_sample)
@overload(np.random.sample)
@overload(np.random.ranf)
def random_impl0():

    @intrinsic
    def _impl(typingcontext):

        def codegen(context, builder, sig, args):
            state_ptr = get_state_ptr(context, builder, 'np')
            return get_next_double(context, builder, state_ptr)
        return (signature(types.float64), codegen)
    return lambda: _impl()