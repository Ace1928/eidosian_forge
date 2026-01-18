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
def subsort(self, other):
    return is_bv_sort(other) and self.size() < other.size()