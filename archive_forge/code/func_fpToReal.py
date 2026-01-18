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
def fpToReal(x, ctx=None):
    """Create a Z3 floating-point conversion expression, from floating-point expression to real.

    >>> x = FP('x', FPSort(8, 24))
    >>> y = fpToReal(x)
    >>> print(is_fp(x))
    True
    >>> print(is_real(y))
    True
    >>> print(is_fp(y))
    False
    >>> print(is_real(x))
    False
    """
    if z3_debug():
        _z3_assert(is_fp(x), 'First argument must be a Z3 floating-point expression')
    ctx = _get_ctx(ctx)
    return ArithRef(Z3_mk_fpa_to_real(ctx.ref(), x.ast), ctx)