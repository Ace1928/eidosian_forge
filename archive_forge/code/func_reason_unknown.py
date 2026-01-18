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
def reason_unknown(self):
    """Return a string that describes why the last `check()` returned `unknown`."""
    return Z3_optimize_get_reason_unknown(self.ctx.ref(), self.optimize)