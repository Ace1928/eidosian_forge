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
def is_string_value(self):
    return Z3_is_string(self.ctx_ref(), self.as_ast())