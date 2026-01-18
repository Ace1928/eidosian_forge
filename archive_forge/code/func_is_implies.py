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
def is_implies(a):
    """Return `True` if `a` is a Z3 implication expression.

    >>> p, q = Bools('p q')
    >>> is_implies(Implies(p, q))
    True
    >>> is_implies(And(p, q))
    False
    """
    return is_app_of(a, Z3_OP_IMPLIES)