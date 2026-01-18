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
@overload(random.gauss)
@overload(random.normalvariate)
def gauss_impl(mu, sigma):
    if isinstance(mu, (types.Float, types.Integer)) and isinstance(sigma, (types.Float, types.Integer)):

        @intrinsic
        def _impl(typingcontext, mu, sigma):
            loc_preprocessor = _double_preprocessor(mu)
            scale_preprocessor = _double_preprocessor(sigma)
            return (signature(types.float64, mu, sigma), _gauss_impl('py', loc_preprocessor, scale_preprocessor))
        return lambda mu, sigma: _impl(mu, sigma)