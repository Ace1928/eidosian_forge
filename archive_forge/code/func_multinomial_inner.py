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
@register_jitable
def multinomial_inner(n, pvals, out):
    fl = out.flat
    sz = out.size
    plen = len(pvals)
    for i in range(0, sz, plen):
        p_sum = 1.0
        n_experiments = n
        for j in range(0, plen - 1):
            p_j = pvals[j]
            n_j = fl[i + j] = np.random.binomial(n_experiments, p_j / p_sum)
            n_experiments -= n_j
            if n_experiments <= 0:
                break
            p_sum -= p_j
        if n_experiments > 0:
            fl[i + plen - 1] = n_experiments