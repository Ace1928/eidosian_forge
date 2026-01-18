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
def help_simplify():
    """Return a string describing all options available for Z3 `simplify` procedure."""
    print(Z3_simplify_get_help(main_ctx().ref()))