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
def fpUnsignedToFP(rm, v, sort, ctx=None):
    """Create a Z3 floating-point conversion expression that represents the
    conversion from an unsigned bit-vector term (encoding an integer) to a floating-point term.

    >>> x_signed = BitVecVal(-5, BitVecSort(32))
    >>> x_fp = fpUnsignedToFP(RNE(), x_signed, Float32())
    >>> x_fp
    fpToFPUnsigned(RNE(), 4294967291)
    >>> simplify(x_fp)
    1*(2**32)
    """
    _z3_assert(is_fprm(rm), 'First argument must be a Z3 floating-point rounding mode expression.')
    _z3_assert(is_bv(v), 'Second argument must be a Z3 bit-vector expression')
    _z3_assert(is_fp_sort(sort), 'Third argument must be a Z3 floating-point sort.')
    ctx = _get_ctx(ctx)
    return FPRef(Z3_mk_fpa_to_fp_unsigned(ctx.ref(), rm.ast, v.ast, sort.ast), ctx)