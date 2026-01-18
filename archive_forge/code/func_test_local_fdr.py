import pytest
import numpy as np
from numpy.testing import (assert_almost_equal, assert_equal,
from statsmodels.stats.multitest import (multipletests, fdrcorrection,
from statsmodels.stats.multicomp import tukeyhsd
from scipy.stats.distributions import norm
import scipy
from packaging import version
def test_local_fdr():
    grid = np.linspace(0.001, 0.999, 1000)
    z0 = norm.ppf(grid)
    z1 = np.linspace(3, 4, 20)
    zs = np.concatenate((z0, z1))
    f1 = np.exp(-z1 ** 2 / 2) / np.sqrt(2 * np.pi)
    r = len(z1) / float(len(z0) + len(z1))
    f1 /= (1 - r) * f1 + r
    for alpha in (None, 0, 1e-08):
        if alpha is None:
            fdr = local_fdr(zs)
        else:
            fdr = local_fdr(zs, alpha=alpha)
        fdr1 = fdr[len(z0):]
        assert_allclose(f1, fdr1, rtol=0.05, atol=0.1)