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
def as_binary_string(self):
    return Z3_get_numeral_binary_string(self.ctx_ref(), self.as_ast())