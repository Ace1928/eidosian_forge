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
@overload(random.randrange)
def randrange_impl_2(start, stop):
    if isinstance(start, types.Integer) and isinstance(stop, types.Integer):
        return lambda start, stop: random.randrange(start, stop, 1)