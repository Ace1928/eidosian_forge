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
def f32_as_int32(builder, val):
    """
    Bitcast a float into a 32-bit integer.
    """
    assert val.type == llvmlite.ir.FloatType()
    return builder.bitcast(val, llvmlite.ir.IntType(32))