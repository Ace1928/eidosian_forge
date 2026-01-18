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
def exponent_as_long(self, biased=True):
    ptr = (ctypes.c_longlong * 1)()
    if not Z3_fpa_get_numeral_exponent_int64(self.ctx.ref(), self.as_ast(), ptr, biased):
        raise Z3Exception('error retrieving the exponent of a numeral.')
    return ptr[0]