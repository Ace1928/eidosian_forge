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
def is_finite_domain(a):
    """Return `True` if `a` is a Z3 finite-domain expression.

    >>> s = FiniteDomainSort('S', 100)
    >>> b = Const('b', s)
    >>> is_finite_domain(b)
    True
    >>> is_finite_domain(Int('x'))
    False
    """
    return isinstance(a, FiniteDomainRef)