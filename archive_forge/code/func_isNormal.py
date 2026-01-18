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
def isNormal(self):
    return Z3_fpa_is_numeral_normal(self.ctx.ref(), self.as_ast())