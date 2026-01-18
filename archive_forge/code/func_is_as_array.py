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
def is_as_array(n):
    """Return true if n is a Z3 expression of the form (_ as-array f)."""
    return isinstance(n, ExprRef) and Z3_is_as_array(n.ctx.ref(), n.as_ast())