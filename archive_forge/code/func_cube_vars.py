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
def cube_vars(self):
    """Access the set of variables that were touched by the most recently generated cube.
        This set of variables can be used as a starting point for additional cubes.
        The idea is that variables that appear in clauses that are reduced by the most recent
        cube are likely more useful to cube on."""
    return self.cube_vs