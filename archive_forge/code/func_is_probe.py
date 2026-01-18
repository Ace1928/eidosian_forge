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
def is_probe(p):
    """Return `True` if `p` is a Z3 probe.

    >>> is_probe(Int('x'))
    False
    >>> is_probe(Probe('memory'))
    True
    """
    return isinstance(p, Probe)