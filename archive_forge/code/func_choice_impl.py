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
def choice_impl(a, size=None, replace=True):
    """
            choice() implementation returning an array of samples
            """
    n = get_source_size(a)
    if replace:
        out = np.empty(size, dtype)
        fl = out.flat
        for i in range(len(fl)):
            j = np.random.randint(0, n)
            fl[i] = getitem(a, j)
        return out
    else:
        out = np.empty(size, dtype)
        if out.size > n:
            raise ValueError("Cannot take a larger sample than population when 'replace=False'")
        permuted_a = np.random.permutation(a)
        fl = out.flat
        for i in range(len(fl)):
            fl[i] = permuted_a[i]
        return out