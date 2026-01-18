import warnings
import numpy as np
from . import _fitpack
from numpy import (atleast_1d, array, ones, zeros, sqrt, ravel, transpose,
from . import dfitpack
def splint(a, b, tck, full_output=0):
    t, c, k = tck
    try:
        c[0][0]
        parametric = True
    except Exception:
        parametric = False
    if parametric:
        return list(map(lambda c, a=a, b=b, t=t, k=k: splint(a, b, [t, c, k]), c))
    else:
        aint, wrk = dfitpack.splint(t, c, k, a, b)
        if full_output:
            return (aint, wrk)
        else:
            return aint