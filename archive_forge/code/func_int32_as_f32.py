import math
import operator
import sys
import numpy as np
import llvmlite.ir
from llvmlite.ir import Constant
from numba.core.imputils import Registry, impl_ret_untracked
from numba import typeof
from numba.core import types, utils, config, cgutils
from numba.core.extending import overload
from numba.core.typing import signature
from numba.cpython.unsafe.numbers import trailing_zeros
def int32_as_f32(builder, val):
    """
    Bitcast a 32-bit integer into a float.
    """
    assert val.type == llvmlite.ir.IntType(32)
    return builder.bitcast(val, llvmlite.ir.FloatType())