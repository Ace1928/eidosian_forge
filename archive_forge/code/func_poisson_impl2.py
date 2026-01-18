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
@overload(np.random.poisson)
def poisson_impl2(lam, size):
    if isinstance(lam, (types.Float, types.Integer)) and is_nonelike(size):
        return lambda lam, size: np.random.poisson(lam)
    if isinstance(lam, (types.Float, types.Integer)) and (isinstance(size, types.Integer) or (isinstance(size, types.UniTuple) and isinstance(size.dtype, types.Integer))):

        def _impl(lam, size):
            out = np.empty(size, dtype=np.intp)
            out_flat = out.flat
            for idx in range(out.size):
                out_flat[idx] = np.random.poisson(lam)
            return out
        return _impl