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
def is_arith_sort(s):
    """Return `True` if s is an arithmetical sort (type).

    >>> is_arith_sort(IntSort())
    True
    >>> is_arith_sort(RealSort())
    True
    >>> is_arith_sort(BoolSort())
    False
    >>> n = Int('x') + 1
    >>> is_arith_sort(n.sort())
    True
    """
    return isinstance(s, ArithSortRef)