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
def qid(self):
    """Return the quantifier id of `self`.
        """
    return _symbol2py(self.ctx, Z3_get_quantifier_id(self.ctx_ref(), self.ast))