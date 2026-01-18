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
def args2params(arguments, keywords, ctx=None):
    """Convert python arguments into a Z3_params object.
    A ':' is added to the keywords, and '_' is replaced with '-'

    >>> args2params(['model', True, 'relevancy', 2], {'elim_and' : True})
    (params model true relevancy 2 elim_and true)
    """
    if z3_debug():
        _z3_assert(len(arguments) % 2 == 0, 'Argument list must have an even number of elements.')
    prev = None
    r = ParamsRef(ctx)
    for a in arguments:
        if prev is None:
            prev = a
        else:
            r.set(prev, a)
            prev = None
    for k in keywords:
        v = keywords[k]
        r.set(k, v)
    return r