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
def fpNEQ(a, b, ctx=None):
    """Create the Z3 floating-point expression `Not(fpEQ(other, self))`.

    >>> x, y = FPs('x y', FPSort(8, 24))
    >>> fpNEQ(x, y)
    Not(fpEQ(x, y))
    >>> (x != y).sexpr()
    '(distinct x y)'
    """
    return Not(fpEQ(a, b, ctx))