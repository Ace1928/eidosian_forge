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
def fpNeg(a, ctx=None):
    """Create a Z3 floating-point addition expression.

    >>> s = FPSort(8, 24)
    >>> rm = RNE()
    >>> x = FP('x', s)
    >>> fpNeg(x)
    -x
    >>> fpNeg(x).sort()
    FPSort(8, 24)
    """
    ctx = _get_ctx(ctx)
    [a] = _coerce_fp_expr_list([a], ctx)
    return FPRef(Z3_mk_fpa_neg(ctx.ref(), a.as_ast()), ctx)