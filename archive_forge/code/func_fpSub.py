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
def fpSub(rm, a, b, ctx=None):
    """Create a Z3 floating-point subtraction expression.

    >>> s = FPSort(8, 24)
    >>> rm = RNE()
    >>> x = FP('x', s)
    >>> y = FP('y', s)
    >>> fpSub(rm, x, y)
    x - y
    >>> fpSub(rm, x, y).sort()
    FPSort(8, 24)
    """
    return _mk_fp_bin(Z3_mk_fpa_sub, rm, a, b, ctx)