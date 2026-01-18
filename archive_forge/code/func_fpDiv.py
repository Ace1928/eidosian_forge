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
def fpDiv(rm, a, b, ctx=None):
    """Create a Z3 floating-point division expression.

    >>> s = FPSort(8, 24)
    >>> rm = RNE()
    >>> x = FP('x', s)
    >>> y = FP('y', s)
    >>> fpDiv(rm, x, y)
    x / y
    >>> fpDiv(rm, x, y).sort()
    FPSort(8, 24)
    """
    return _mk_fp_bin(Z3_mk_fpa_div, rm, a, b, ctx)