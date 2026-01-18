import numpy as np
from numpy import poly1d, sqrt, exp
import scipy
from scipy import stats, special
from scipy.stats import distributions
from statsmodels.stats.moment_helpers import mvsk2mc, mc2mvsk
def pdf_moments_st(cnt):
    """Return the Gaussian expanded pdf function given the list of central
    moments (first one is mean).

    version of scipy.stats, any changes ?
    the scipy.stats version has a bug and returns normal distribution
    """
    N = len(cnt)
    if N < 2:
        raise ValueError('At least two moments must be given to approximate the pdf.')
    totp = poly1d(1)
    sig = sqrt(cnt[1])
    mu = cnt[0]
    if N > 2:
        Dvals = _hermnorm(N + 1)
    for k in range(3, N + 1):
        Ck = 0.0
        for n in range((k - 3) / 2):
            m = k - 2 * n
            if m % 2:
                momdiff = cnt[m - 1]
            else:
                momdiff = cnt[m - 1] - sig * sig * scipy.factorial2(m - 1)
            Ck += Dvals[k][m] / sig ** m * momdiff
        raise SystemError
        print(Dvals)
        print(Ck)
        totp = totp + Ck * Dvals[k]

    def thisfunc(x):
        xn = (x - mu) / sig
        return totp(xn) * exp(-xn * xn / 2.0) / sqrt(2 * np.pi) / sig
    return (thisfunc, totp)