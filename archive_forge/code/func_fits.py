import sys
import io
from .z3consts import *
from .z3core import *
from ctypes import *
def fits(f, space_left):
    s, nl = f.space_upto_nl()
    return s <= space_left