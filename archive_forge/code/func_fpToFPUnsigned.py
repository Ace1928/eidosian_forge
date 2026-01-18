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
def fpToFPUnsigned(rm, x, s, ctx=None):
    """Create a Z3 floating-point conversion expression, from unsigned bit-vector to floating-point expression."""
    if z3_debug():
        _z3_assert(is_fprm(rm), 'First argument must be a Z3 floating-point rounding mode expression')
        _z3_assert(is_bv(x), 'Second argument must be a Z3 bit-vector expression')
        _z3_assert(is_fp_sort(s), 'Third argument must be Z3 floating-point sort')
    ctx = _get_ctx(ctx)
    return FPRef(Z3_mk_fpa_to_fp_unsigned(ctx.ref(), rm.ast, x.ast, s.ast), ctx)