import numpy as np
from collections import namedtuple
from scipy import special
from scipy import stats
from ._axis_nan_policy import _axis_nan_policy_factory
def pmf_recursive(self, k, m, n):
    """Probability mass function, recursive version"""
    self._resize_fmnks(m, n, np.max(k))
    for i in np.ravel(k):
        self._f(m, n, i)
    return self._fmnks[m, n, k] / special.binom(m + n, m)