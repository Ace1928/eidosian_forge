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
def conflict(self, deps=[], eqs=[]):
    self.propagate(BoolVal(False, self.ctx()), deps, eqs)