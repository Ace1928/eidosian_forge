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
def np_uniform_impl3(low, high, size):
    if isinstance(low, (types.Float, types.Integer)) and isinstance(high, (types.Float, types.Integer)) and is_nonelike(size):
        return lambda low, high, size: np.random.uniform(low, high)
    if isinstance(low, (types.Float, types.Integer)) and isinstance(high, (types.Float, types.Integer)) and (isinstance(size, types.Integer) or (isinstance(size, types.UniTuple) and isinstance(size.dtype, types.Integer))):

        def _impl(low, high, size):
            out = np.empty(size)
            out_flat = out.flat
            for idx in range(out.size):
                out_flat[idx] = np.random.uniform(low, high)
            return out
        return _impl