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
def reset_params():
    """Reset all global (or module) parameters.
    """
    Z3_global_param_reset_all()