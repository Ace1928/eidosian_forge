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
def tactics(ctx=None):
    """Return a list of all available tactics in Z3.

    >>> l = tactics()
    >>> l.count('simplify') == 1
    True
    """
    ctx = _get_ctx(ctx)
    return [Z3_get_tactic_name(ctx.ref(), i) for i in range(Z3_get_num_tactics(ctx.ref()))]