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
def fpPlusZero(s):
    """Create a Z3 floating-point +0.0 term."""
    _z3_assert(isinstance(s, FPSortRef), 'sort mismatch')
    return FPNumRef(Z3_mk_fpa_zero(s.ctx_ref(), s.ast, False), s.ctx)