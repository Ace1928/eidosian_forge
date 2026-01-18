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
def is_seq(a):
    """Return `True` if `a` is a Z3 sequence expression.
    >>> print (is_seq(Unit(IntVal(0))))
    True
    >>> print (is_seq(StringVal("abc")))
    True
    """
    return isinstance(a, SeqRef)