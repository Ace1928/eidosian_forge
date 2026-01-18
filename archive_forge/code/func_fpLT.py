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
def fpLT(a, b, ctx=None):
    """Create the Z3 floating-point expression `other < self`.

    >>> x, y = FPs('x y', FPSort(8, 24))
    >>> fpLT(x, y)
    x < y
    >>> (x < y).sexpr()
    '(fp.lt x y)'
    """
    return _mk_fp_bin_pred(Z3_mk_fpa_lt, a, b, ctx)