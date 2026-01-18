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
def is_to_int(a):
    """Return `True` if `a` is an expression of the form ToInt(b).

    >>> x = Real('x')
    >>> n = ToInt(x)
    >>> n
    ToInt(x)
    >>> is_to_int(n)
    True
    >>> is_to_int(x)
    False
    """
    return is_app_of(a, Z3_OP_TO_INT)