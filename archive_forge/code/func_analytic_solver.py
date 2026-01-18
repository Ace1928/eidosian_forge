from collections import OrderedDict
from functools import reduce, partial
from itertools import chain
from operator import attrgetter, mul
import math
import warnings
from ..units import (
from ..util.pyutil import deprecated
from ..util._expr import Expr, Symbol
from .rates import RateExpr, MassAction
def analytic_solver(x0, y0, p0, be):
    if preferred is None:
        _preferred = None
    else:
        _preferred = list(preferred)
    A = be.Matrix(compo_vecs)
    rA, pivots = A.rref()
    analytic_exprs = OrderedDict()
    for ri, ci1st in enumerate(pivots):
        for idx in range(ci1st, odesys.ny):
            key = odesys.names[idx]
            if rA[ri, idx] == 0:
                continue
            if _preferred is None or key in _preferred:
                terms = [rA[ri, di] * (odesys.dep[di] - y0[odesys.dep[di]]) for di in range(ci1st, odesys.ny) if di != idx]
                analytic_exprs[odesys[key]] = y0[odesys.dep[idx]] - sum(terms) / rA[ri, idx]
                if _preferred is not None:
                    _preferred.remove(key)
                break
    for k in reversed(list(analytic_exprs.keys())):
        analytic_exprs[k] = analytic_exprs[k].subs(analytic_exprs)
    if _preferred is not None and len(_preferred) > 0:
        raise ValueError('Failed to obtain analytic expression for: %s' % ', '.join(_preferred))
    return analytic_exprs