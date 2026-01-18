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
def fpBVToFP(v, sort, ctx=None):
    """Create a Z3 floating-point conversion expression that represents the
    conversion from a bit-vector term to a floating-point term.

    >>> x_bv = BitVecVal(0x3F800000, 32)
    >>> x_fp = fpBVToFP(x_bv, Float32())
    >>> x_fp
    fpToFP(1065353216)
    >>> simplify(x_fp)
    1
    """
    _z3_assert(is_bv(v), 'First argument must be a Z3 bit-vector expression')
    _z3_assert(is_fp_sort(sort), 'Second argument must be a Z3 floating-point sort.')
    ctx = _get_ctx(ctx)
    return FPRef(Z3_mk_fpa_to_fp_bv(ctx.ref(), v.ast, sort.ast), ctx)