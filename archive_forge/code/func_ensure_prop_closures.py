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
def ensure_prop_closures():
    global _prop_closures
    if _prop_closures is None:
        _prop_closures = PropClosures()