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
def fpGEQ(a, b, ctx=None):
    """Create the Z3 floating-point expression `other >= self`.

    >>> x, y = FPs('x y', FPSort(8, 24))
    >>> fpGEQ(x, y)
    x >= y
    >>> (x >= y).sexpr()
    '(fp.geq x y)'
    """
    return _mk_fp_bin_pred(Z3_mk_fpa_geq, a, b, ctx)