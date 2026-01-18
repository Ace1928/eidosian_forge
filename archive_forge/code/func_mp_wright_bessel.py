from functools import lru_cache
import os
from time import time
import numpy as np
from scipy.special._mptestutils import mpf2float
def mp_wright_bessel(a, b, x, dps=50, maxterms=2000):
    """Compute Wright's generalized Bessel function as Series with mpmath.
    """
    with mp.workdps(dps):
        a, b, x = (mp.mpf(a), mp.mpf(b), mp.mpf(x))
        res = mp.nsum(lambda k: x ** k / mp.fac(k) * rgamma_cached(a * k + b, dps=dps), [0, mp.inf], tol=dps, method='s', steps=[maxterms])
        return mpf2float(res)