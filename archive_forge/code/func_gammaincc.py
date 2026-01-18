import os
from time import time
import numpy as np
from numpy import pi
from scipy.special._mptestutils import mpf2float
def gammaincc(a, x, dps=50, maxterms=10 ** 8):
    """Compute gammaincc exactly like mpmath does but allow for more
    terms in hypercomb. See

    mpmath/functions/expintegrals.py#L187

    in the mpmath github repository.

    """
    with mp.workdps(dps):
        z, a = (a, x)
        if mp.isint(z):
            try:
                return mpf2float(mp.gammainc(z, a=a, regularized=True))
            except mp.libmp.NoConvergence:
                pass
        nega = mp.fneg(a, exact=True)
        G = [z]
        try:

            def h(z):
                r = z - 1
                return [([mp.exp(nega), a], [1, r], [], G, [1, -r], [], 1 / nega)]
            return mpf2float(mp.hypercomb(h, [z], force_series=True))
        except mp.libmp.NoConvergence:

            def h(z):
                T1 = ([], [1, z - 1], [z], G, [], [], 0)
                T2 = ([-mp.exp(nega), a, z], [1, z, -1], [], G, [1], [1 + z], a)
                return (T1, T2)
            return mpf2float(mp.hypercomb(h, [z], maxterms=maxterms))