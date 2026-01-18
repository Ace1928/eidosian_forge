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
def get_shifted_int(nbits):
    shift = builder.sub(c32, nbits)
    y = get_next_int32(context, builder, state_ptr)
    if nbits.type.width < y.type.width:
        shift = builder.zext(shift, y.type)
    elif nbits.type.width > y.type.width:
        shift = builder.trunc(shift, y.type)
    if is_numpy:
        mask = builder.not_(ir.Constant(y.type, 0))
        mask = builder.lshr(mask, shift)
        return builder.and_(y, mask)
    else:
        return builder.lshr(y, shift)