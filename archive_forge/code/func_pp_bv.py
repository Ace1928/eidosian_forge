import sys
import io
from .z3consts import *
from .z3core import *
from ctypes import *
def pp_bv(self, a):
    return to_format(a.as_string())