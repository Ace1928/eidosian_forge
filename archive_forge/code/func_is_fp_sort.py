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
def is_fp_sort(s):
    """Return True if `s` is a Z3 floating-point sort.

    >>> is_fp_sort(FPSort(8, 24))
    True
    >>> is_fp_sort(IntSort())
    False
    """
    return isinstance(s, FPSortRef)