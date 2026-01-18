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
def do_shuffle_impl(x, rng):
    if not isinstance(x, types.Buffer):
        raise TypeError('The argument to shuffle() should be a buffer type')
    if rng == 'np':
        rand = np.random.randint
    elif rng == 'py':
        rand = random.randrange
    if x.ndim == 1:

        def impl(x):
            i = x.shape[0] - 1
            while i > 0:
                j = rand(i + 1)
                x[i], x[j] = (x[j], x[i])
                i -= 1
    else:

        def impl(x):
            i = x.shape[0] - 1
            while i > 0:
                j = rand(i + 1)
                x[i], x[j] = (np.copy(x[j]), np.copy(x[i]))
                i -= 1
    return impl