import sys
import io
from .z3consts import *
from .z3core import *
from ctypes import *
def pp_algebraic(self, a):
    return to_format(a.as_decimal(self.precision))