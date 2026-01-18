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
@overload(np.random.uniform)
def np_uniform_impl2(low, high):
    if isinstance(low, (types.Float, types.Integer)) and isinstance(high, (types.Float, types.Integer)):

        @intrinsic
        def _impl(typingcontext, low, high):
            low_preprocessor = _double_preprocessor(low)
            high_preprocessor = _double_preprocessor(high)
            return (signature(types.float64, low, high), uniform_impl('np', low_preprocessor, high_preprocessor))
        return lambda low, high: _impl(low, high)