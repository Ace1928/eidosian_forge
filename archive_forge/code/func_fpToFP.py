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
def fpToFP(a1, a2=None, a3=None, ctx=None):
    """Create a Z3 floating-point conversion expression from other term sorts
    to floating-point.

    From a bit-vector term in IEEE 754-2008 format:
    >>> x = FPVal(1.0, Float32())
    >>> x_bv = fpToIEEEBV(x)
    >>> simplify(fpToFP(x_bv, Float32()))
    1

    From a floating-point term with different precision:
    >>> x = FPVal(1.0, Float32())
    >>> x_db = fpToFP(RNE(), x, Float64())
    >>> x_db.sort()
    FPSort(11, 53)

    From a real term:
    >>> x_r = RealVal(1.5)
    >>> simplify(fpToFP(RNE(), x_r, Float32()))
    1.5

    From a signed bit-vector term:
    >>> x_signed = BitVecVal(-5, BitVecSort(32))
    >>> simplify(fpToFP(RNE(), x_signed, Float32()))
    -1.25*(2**2)
    """
    ctx = _get_ctx(ctx)
    if is_bv(a1) and is_fp_sort(a2):
        return FPRef(Z3_mk_fpa_to_fp_bv(ctx.ref(), a1.ast, a2.ast), ctx)
    elif is_fprm(a1) and is_fp(a2) and is_fp_sort(a3):
        return FPRef(Z3_mk_fpa_to_fp_float(ctx.ref(), a1.ast, a2.ast, a3.ast), ctx)
    elif is_fprm(a1) and is_real(a2) and is_fp_sort(a3):
        return FPRef(Z3_mk_fpa_to_fp_real(ctx.ref(), a1.ast, a2.ast, a3.ast), ctx)
    elif is_fprm(a1) and is_bv(a2) and is_fp_sort(a3):
        return FPRef(Z3_mk_fpa_to_fp_signed(ctx.ref(), a1.ast, a2.ast, a3.ast), ctx)
    else:
        raise Z3Exception('Unsupported combination of arguments for conversion to floating-point term.')