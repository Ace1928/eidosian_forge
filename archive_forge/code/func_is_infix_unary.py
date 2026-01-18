import sys
import io
from .z3consts import *
from .z3core import *
from ctypes import *
def is_infix_unary(self, a):
    return self.is_infix(a) or self.is_unary(a)