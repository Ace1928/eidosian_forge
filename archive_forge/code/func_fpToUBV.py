from . import z3core
from .z3core import *
from .z3types import *
from .z3consts import *
from .z3printer import *
from fractions import Fraction
import sys
import io
import math
import copy
def fpToUBV(rm, x, s, ctx=None):
    """Create a Z3 floating-point conversion expression, from floating-point expression to unsigned bit-vector.

    >>> x = FP('x', FPSort(8, 24))
    >>> y = fpToUBV(RTZ(), x, BitVecSort(32))
    >>> print(is_fp(x))
    True
    >>> print(is_bv(y))
    True
    >>> print(is_fp(y))
    False
    >>> print(is_bv(x))
    False
    """
    if z3_debug():
        _z3_assert(is_fprm(rm), 'First argument must be a Z3 floating-point rounding mode expression')
        _z3_assert(is_fp(x), 'Second argument must be a Z3 floating-point expression')
        _z3_assert(is_bv_sort(s), 'Third argument must be Z3 bit-vector sort')
    ctx = _get_ctx(ctx)
    return BitVecRef(Z3_mk_fpa_to_ubv(ctx.ref(), rm.ast, x.ast, s.size()), ctx)