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
def next_split(self, t, idx, phase):
    return Z3_solver_next_split(self.ctx_ref(), ctypes.c_void_p(self.cb), t.ast, idx, phase)