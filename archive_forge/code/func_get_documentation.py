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
def get_documentation(self, n):
    """Return the documentation string of the parameter named `n`.
        """
    return Z3_param_descrs_get_documentation(self.ctx.ref(), self.descr, to_symbol(n, self.ctx))