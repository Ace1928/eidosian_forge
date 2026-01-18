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
def fpSqrt(rm, a, ctx=None):
    """Create a Z3 floating-point square root expression.
    """
    return _mk_fp_unary(Z3_mk_fpa_sqrt, rm, a, ctx)