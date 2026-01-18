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
def negate_real(builder, val):
    """
    Negate real number *val*, with proper handling of zeros.
    """
    return builder.fsub(Constant(val.type, -0.0), val)