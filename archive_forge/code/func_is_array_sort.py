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
def is_array_sort(a):
    return Z3_get_sort_kind(a.ctx.ref(), Z3_get_sort(a.ctx.ref(), a.ast)) == Z3_ARRAY_SORT