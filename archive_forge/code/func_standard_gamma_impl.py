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
@overload(np.random.standard_gamma)
def standard_gamma_impl(shape, size):
    if is_nonelike(size):
        return lambda shape, size: np.random.standard_gamma(shape)
    if isinstance(size, types.Integer) or (isinstance(size, types.UniTuple) and isinstance(size.dtype, types.Integer)):

        def _impl(shape, size):
            out = np.empty(size)
            out_flat = out.flat
            for idx in range(out.size):
                out_flat[idx] = np.random.standard_gamma(shape)
            return out
        return _impl