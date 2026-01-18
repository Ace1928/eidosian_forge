import warnings
import numpy as np
from . import _fitpack
from numpy import (atleast_1d, array, ones, zeros, sqrt, ravel, transpose,
from . import dfitpack
def sproot(tck, mest=10):
    t, c, k = tck
    if k != 3:
        raise ValueError('sproot works only for cubic (k=3) splines')
    try:
        c[0][0]
        parametric = True
    except Exception:
        parametric = False
    if parametric:
        return list(map(lambda c, t=t, k=k, mest=mest: sproot([t, c, k], mest), c))
    else:
        if len(t) < 8:
            raise TypeError('The number of knots %d>=8' % len(t))
        z, m, ier = dfitpack.sproot(t, c, mest)
        if ier == 10:
            raise TypeError('Invalid input data. t1<=..<=t4<t5<..<tn-3<=..<=tn must hold.')
        if ier == 0:
            return z[:m]
        if ier == 1:
            warnings.warn(RuntimeWarning('The number of zeros exceeds mest'), stacklevel=2)
            return z[:m]
        raise TypeError('Unknown error')