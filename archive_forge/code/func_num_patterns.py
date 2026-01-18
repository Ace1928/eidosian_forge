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
def num_patterns(self):
    """Return the number of patterns (i.e., quantifier instantiation hints) in `self`.

        >>> f = Function('f', IntSort(), IntSort())
        >>> g = Function('g', IntSort(), IntSort())
        >>> x = Int('x')
        >>> q = ForAll(x, f(x) != g(x), patterns = [ f(x), g(x) ])
        >>> q.num_patterns()
        2
        """
    return int(Z3_get_quantifier_num_patterns(self.ctx_ref(), self.ast))