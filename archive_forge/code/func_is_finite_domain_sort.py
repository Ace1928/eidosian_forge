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
def is_finite_domain_sort(s):
    """Return True if `s` is a Z3 finite-domain sort.

    >>> is_finite_domain_sort(FiniteDomainSort('S', 100))
    True
    >>> is_finite_domain_sort(IntSort())
    False
    """
    return isinstance(s, FiniteDomainSortRef)