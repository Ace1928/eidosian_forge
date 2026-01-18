import sys
import io
from .z3consts import *
from .z3core import *
from ctypes import *
def space_upto_nl(self):
    return (getattr(self, 'size', len(self.string)), False)