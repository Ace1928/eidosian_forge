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
def add_sort(self, sort):
    Z3_parser_context_add_sort(self.ctx.ref(), self.pctx, sort.as_ast())