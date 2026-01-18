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
def set_default_fp_sort(ebits, sbits, ctx=None):
    global _dflt_fpsort_ebits
    global _dflt_fpsort_sbits
    _dflt_fpsort_ebits = ebits
    _dflt_fpsort_sbits = sbits