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
def domain_n(self, i):
    """Shorthand for self.sort().domain_n(i)`."""
    return self.sort().domain_n(i)