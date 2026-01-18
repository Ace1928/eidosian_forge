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
def max_euler_step_cb(x, y, p=()):
    _x, _y, _p = odesys.pre_process(*odesys.to_arrays(x, y, p))
    upper_bounds = rsys.upper_conc_bounds(_y)
    fvec = odesys.f_cb(_x[0], _y, _p)
    h = []
    for idx, fcomp in enumerate(fvec):
        if fcomp == 0:
            h.append(float('inf'))
        elif fcomp > 0:
            h.append((upper_bounds[idx] - _y[idx]) / fcomp)
        else:
            h.append(-_y[idx] / fcomp)
    min_h = min(h)
    return min(min_h, 1)