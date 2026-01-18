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
def get_rule_names_along_trace(self):
    """retrieve rule names along the counterexample trace"""
    names = _symbol2py(self.ctx, Z3_fixedpoint_get_rule_names_along_trace(self.ctx.ref(), self.fixedpoint))
    return names.split(';')