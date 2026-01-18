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
def set_predicate_representation(self, f, *representations):
    """Control how relation is represented"""
    representations = _get_args(representations)
    representations = [to_symbol(s) for s in representations]
    sz = len(representations)
    args = (Symbol * sz)()
    for i in range(sz):
        args[i] = representations[i]
    Z3_fixedpoint_set_predicate_representation(self.ctx.ref(), self.fixedpoint, f.ast, sz, args)