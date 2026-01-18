import numpy as np
from collections import namedtuple
from scipy import special
from scipy import stats
from ._axis_nan_policy import _axis_nan_policy_factory
def pmf_iterative(self, k, m, n):
    """Probability mass function, iterative version"""
    fmnks = {}
    for i in np.ravel(k):
        fmnks = _mwu_f_iterative(m, n, i, fmnks)
    return np.array([fmnks[m, n, ki] for ki in k]) / special.binom(m + n, m)