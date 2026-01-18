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
@overload(random.paretovariate)
def paretovariate_impl(alpha):
    if isinstance(alpha, types.Float):

        def _impl(alpha):
            """Pareto distribution.  Taken from CPython."""
            u = 1.0 - random.random()
            return 1.0 / u ** (1.0 / alpha)
        return _impl