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
@overload(np.random.rayleigh)
def rayleigh_impl1(scale):
    if isinstance(scale, (types.Float, types.Integer)):

        def impl(scale):
            if scale <= 0.0:
                raise ValueError('rayleigh(): scale <= 0')
            return scale * math.sqrt(-2.0 * math.log(1.0 - np.random.random()))
        return impl