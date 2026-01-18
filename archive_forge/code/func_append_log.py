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
def append_log(s):
    """Append user-defined string to interaction log. """
    Z3_append_log(s)