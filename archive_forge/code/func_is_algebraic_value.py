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
def is_algebraic_value(a):
    """Return `True` if `a` is an algebraic value of sort Real.

    >>> is_algebraic_value(RealVal("3/5"))
    False
    >>> n = simplify(Sqrt(2))
    >>> n
    1.4142135623?
    >>> is_algebraic_value(n)
    True
    """
    return is_arith(a) and a.is_real() and _is_algebraic(a.ctx, a.as_ast())