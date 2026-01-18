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
def lower_values(self, obj):
    if not isinstance(obj, OptimizeObjective):
        raise Z3Exception('Expecting objective handle returned by maximize/minimize')
    return obj.lower_values()