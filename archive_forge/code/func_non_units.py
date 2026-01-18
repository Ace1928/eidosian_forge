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
def non_units(self):
    """Return an AST vector containing all atomic formulas in solver state that are not units.
        """
    return AstVector(Z3_solver_get_non_units(self.ctx.ref(), self.solver), self.ctx)