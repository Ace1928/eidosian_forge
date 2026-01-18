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
def is_app_of(a, k):
    """Return `True` if `a` is an application of the given kind `k`.

    >>> x = Int('x')
    >>> n = x + 1
    >>> is_app_of(n, Z3_OP_ADD)
    True
    >>> is_app_of(n, Z3_OP_MUL)
    False
    """
    return is_app(a) and a.decl().kind() == k