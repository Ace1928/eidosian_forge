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
def fpFP(sgn, exp, sig, ctx=None):
    """Create the Z3 floating-point value `fpFP(sgn, sig, exp)` from the three bit-vectors sgn, sig, and exp.

    >>> s = FPSort(8, 24)
    >>> x = fpFP(BitVecVal(1, 1), BitVecVal(2**7-1, 8), BitVecVal(2**22, 23))
    >>> print(x)
    fpFP(1, 127, 4194304)
    >>> xv = FPVal(-1.5, s)
    >>> print(xv)
    -1.5
    >>> slvr = Solver()
    >>> slvr.add(fpEQ(x, xv))
    >>> slvr.check()
    sat
    >>> xv = FPVal(+1.5, s)
    >>> print(xv)
    1.5
    >>> slvr = Solver()
    >>> slvr.add(fpEQ(x, xv))
    >>> slvr.check()
    unsat
    """
    _z3_assert(is_bv(sgn) and is_bv(exp) and is_bv(sig), 'sort mismatch')
    _z3_assert(sgn.sort().size() == 1, 'sort mismatch')
    ctx = _get_ctx(ctx)
    _z3_assert(ctx == sgn.ctx == exp.ctx == sig.ctx, 'context mismatch')
    return FPRef(Z3_mk_fpa_fp(ctx.ref(), sgn.ast, exp.ast, sig.ast), ctx)