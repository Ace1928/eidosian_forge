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
def is_int_value(self):
    return self.denominator().is_int() and self.denominator_as_long() == 1