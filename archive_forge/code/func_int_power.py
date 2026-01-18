import math
import numbers
import numpy as np
import operator
from llvmlite import ir
from llvmlite.ir import Constant
from numba.core.imputils import (lower_builtin, lower_getattr,
from numba.core import typing, types, utils, errors, cgutils, optional
from numba.core.extending import intrinsic, overload_method
from numba.cpython.unsafe.numbers import viewer
def int_power(a, b):
    r = tp(1)
    a = tp(a)
    if b < 0:
        invert = True
        exp = -b
        if exp < 0:
            raise OverflowError
        if is_integer:
            if a == 0:
                if zerodiv_return:
                    return zerodiv_return
                else:
                    raise ZeroDivisionError('0 cannot be raised to a negative power')
            if a != 1 and a != -1:
                return 0
    else:
        invert = False
        exp = b
    if exp > 65536:
        return math.pow(a, float(b))
    while exp != 0:
        if exp & 1:
            r *= a
        exp >>= 1
        a *= a
    return 1.0 / r if invert else r