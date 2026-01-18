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
@overload(np.random.lognormal)
def np_log_normal_impl2(mean, sigma):
    if isinstance(mean, (types.Float, types.Integer)) and isinstance(sigma, (types.Float, types.Integer)):
        fn = register_jitable(_lognormvariate_impl(np.random.normal))
        return lambda mean, sigma: fn(mean, sigma)