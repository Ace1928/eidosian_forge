import math
from bisect import bisect
from .backend import xrange
from .backend import MPZ, MPZ_ZERO, MPZ_ONE, MPZ_TWO, MPZ_FIVE, BACKEND
from .libmpf import (
from .libintmath import ifib
def constant_memo(f):
    """
    Decorator for caching computed values of mathematical
    constants. This decorator should be applied to a
    function taking a single argument prec as input and
    returning a fixed-point value with the given precision.
    """
    f.memo_prec = -1
    f.memo_val = None

    def g(prec, **kwargs):
        memo_prec = f.memo_prec
        if prec <= memo_prec:
            return f.memo_val >> memo_prec - prec
        newprec = int(prec * 1.05 + 10)
        f.memo_val = f(newprec, **kwargs)
        f.memo_prec = newprec
        return f.memo_val >> newprec - prec
    g.__name__ = f.__name__
    g.__doc__ = f.__doc__
    return g