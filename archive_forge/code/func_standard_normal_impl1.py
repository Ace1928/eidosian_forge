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
@overload(np.random.standard_normal)
def standard_normal_impl1(size):
    if is_nonelike(size):
        return lambda size: np.random.standard_normal()
    if isinstance(size, types.Integer) or (isinstance(size, types.UniTuple) and isinstance(size.dtype, types.Integer)):

        def _impl(size):
            out = np.empty(size)
            out_flat = out.flat
            for idx in range(out.size):
                out_flat[idx] = np.random.standard_normal()
            return out
        return _impl