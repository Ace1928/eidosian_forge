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
def get_num():
    bbwhile = builder.append_basic_block('while')
    bbend = builder.append_basic_block('while.end')
    builder.branch(bbwhile)
    builder.position_at_end(bbwhile)
    r = get_next_int(context, builder, state_ptr, nbits, state == 'np')
    r = builder.trunc(r, ty)
    too_large = builder.icmp_signed('>=', r, n)
    builder.cbranch(too_large, bbwhile, bbend)
    builder.position_at_end(bbend)
    builder.store(r, rptr)