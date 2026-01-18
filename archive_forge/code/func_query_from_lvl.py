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
def query_from_lvl(self, lvl, *query):
    """Query the fixedpoint engine whether formula is derivable starting at the given query level.
        """
    query = _get_args(query)
    sz = len(query)
    if sz >= 1 and isinstance(query[0], FuncDecl):
        _z3_assert(False, 'unsupported')
    else:
        if sz == 1:
            query = query[0]
        else:
            query = And(query)
        query = self.abstract(query, False)
        r = Z3_fixedpoint_query_from_lvl(self.ctx.ref(), self.fixedpoint, query.as_ast(), lvl)
    return CheckSatResult(r)