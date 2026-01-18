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
def no_pattern(self, idx):
    """Return a no-pattern."""
    if z3_debug():
        _z3_assert(idx < self.num_no_patterns(), 'Invalid no-pattern idx')
    return _to_expr_ref(Z3_get_quantifier_no_pattern_ast(self.ctx_ref(), self.ast, idx), self.ctx)