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
def poisson_impl(lam):
    """Numpy's algorithm for poisson() on small *lam*.

                    This method is invoked only if the parameter lambda of the
                    distribution is small ( < 10 ). The algorithm used is
                    described in "Knuth, D. 1969. 'Seminumerical Algorithms.
                    The Art of Computer Programming' vol 2.
                    """
    if lam < 0.0:
        raise ValueError('poisson(): lambda < 0')
    if lam == 0.0:
        return 0
    enlam = _exp(-lam)
    X = 0
    prod = 1.0
    while 1:
        U = _random()
        prod *= U
        if prod <= enlam:
            return X
        X += 1