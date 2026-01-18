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
def int64_as_f64(builder, val):
    """
    Bitcast a 64-bit integer into a double.
    """
    assert val.type == llvmlite.ir.IntType(64)
    return builder.bitcast(val, llvmlite.ir.DoubleType())