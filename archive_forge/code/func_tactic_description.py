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
def tactic_description(name, ctx=None):
    """Return a short description for the tactic named `name`.

    >>> d = tactic_description('simplify')
    """
    ctx = _get_ctx(ctx)
    return Z3_tactic_get_descr(ctx.ref(), name)