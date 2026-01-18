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
@overload(np.random.standard_t)
def standard_t_impl(df):
    if isinstance(df, (types.Float, types.Integer)):

        def _impl(df):
            N = np.random.standard_normal()
            G = np.random.standard_gamma(df / 2.0)
            X = math.sqrt(df / 2.0) * N / math.sqrt(G)
            return X
        return _impl