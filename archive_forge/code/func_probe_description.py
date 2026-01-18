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
def probe_description(name, ctx=None):
    """Return a short description for the probe named `name`.

    >>> d = probe_description('memory')
    """
    ctx = _get_ctx(ctx)
    return Z3_probe_get_descr(ctx.ref(), name)