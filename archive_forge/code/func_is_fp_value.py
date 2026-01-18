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
def is_fp_value(a):
    """Return `True` if `a` is a Z3 floating-point numeral value.

    >>> b = FP('b', FPSort(8, 24))
    >>> is_fp_value(b)
    False
    >>> b = FPVal(1.0, FPSort(8, 24))
    >>> b
    1
    >>> is_fp_value(b)
    True
    """
    return is_fp(a) and _is_numeral(a.ctx, a.ast)