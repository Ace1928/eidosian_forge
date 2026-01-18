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
def is_finite_domain_value(a):
    """Return `True` if `a` is a Z3 finite-domain value.

    >>> s = FiniteDomainSort('S', 100)
    >>> b = Const('b', s)
    >>> is_finite_domain_value(b)
    False
    >>> b = FiniteDomainVal(10, s)
    >>> b
    10
    >>> is_finite_domain_value(b)
    True
    """
    return is_finite_domain(a) and _is_numeral(a.ctx, a.as_ast())