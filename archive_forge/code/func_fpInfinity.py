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
def fpInfinity(s, negative):
    """Create a Z3 floating-point +oo or -oo term."""
    _z3_assert(isinstance(s, FPSortRef), 'sort mismatch')
    _z3_assert(isinstance(negative, bool), 'expected Boolean flag')
    return FPNumRef(Z3_mk_fpa_inf(s.ctx_ref(), s.ast, negative), s.ctx)