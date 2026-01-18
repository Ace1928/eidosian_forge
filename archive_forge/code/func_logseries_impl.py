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
@overload(np.random.logseries)
def logseries_impl(p, size):
    if is_nonelike(size):
        return lambda p, size: np.random.logseries(p)
    if isinstance(size, types.Integer) or (isinstance(size, types.UniTuple) and isinstance(size.dtype, types.Integer)):

        def _impl(p, size):
            out = np.empty(size, dtype=np.int64)
            out_flat = out.flat
            for idx in range(out.size):
                out_flat[idx] = np.random.logseries(p)
            return out
        return _impl