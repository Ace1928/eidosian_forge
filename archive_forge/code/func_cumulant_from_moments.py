import warnings
import numpy as np
from numpy.polynomial.hermite_e import HermiteE
from scipy.special import factorial
from scipy.stats import rv_continuous
import scipy.special as special
def cumulant_from_moments(momt, n):
    """Compute n-th cumulant given moments.

    Parameters
    ----------
    momt : array_like
        `momt[j]` contains `(j+1)`-th moment.
        These can be raw moments around zero, or central moments
        (in which case, `momt[0]` == 0).
    n : int
        which cumulant to calculate (must be >1)

    Returns
    -------
    kappa : float
        n-th cumulant.
    """
    if n < 1:
        raise ValueError('Expected a positive integer. Got %s instead.' % n)
    if len(momt) < n:
        raise ValueError('%s-th cumulant requires %s moments, only got %s.' % (n, n, len(momt)))
    kappa = 0.0
    for p in _faa_di_bruno_partitions(n):
        r = sum((k for m, k in p))
        term = (-1) ** (r - 1) * factorial(r - 1)
        for m, k in p:
            term *= np.power(momt[m - 1] / factorial(m), k) / factorial(k)
        kappa += term
    kappa *= factorial(n)
    return kappa