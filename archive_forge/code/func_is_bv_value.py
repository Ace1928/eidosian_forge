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
def is_bv_value(a):
    """Return `True` if `a` is a Z3 bit-vector numeral value.

    >>> b = BitVec('b', 32)
    >>> is_bv_value(b)
    False
    >>> b = BitVecVal(10, 32)
    >>> b
    10
    >>> is_bv_value(b)
    True
    """
    return is_bv(a) and _is_numeral(a.ctx, a.as_ast())