import numpy.testing as npt
from numpy.testing import assert_allclose
import numpy as np
import pytest
from scipy import stats
from .common_tests import (check_normalization, check_moment,
from scipy.stats._distr_params import distdiscrete, invdistdiscrete
from scipy.stats._distn_infrastructure import rv_discrete_frozen
def check_pmf_cdf(distfn, arg, distname):
    if hasattr(distfn, 'xk'):
        index = distfn.xk
    else:
        startind = int(distfn.ppf(0.01, *arg) - 1)
        index = list(range(startind, startind + 10))
    cdfs = distfn.cdf(index, *arg)
    pmfs_cum = distfn.pmf(index, *arg).cumsum()
    atol, rtol = (1e-10, 1e-10)
    if distname == 'skellam':
        atol, rtol = (1e-05, 1e-05)
    npt.assert_allclose(cdfs - cdfs[0], pmfs_cum - pmfs_cum[0], atol=atol, rtol=rtol)
    k = np.asarray(index)
    k_shifted = k[:-1] + np.diff(k) / 2
    npt.assert_equal(distfn.pmf(k_shifted, *arg), 0)
    loc = 0.5
    dist = distfn(*arg, loc=loc)
    npt.assert_allclose(dist.pmf(k[1:] + loc), np.diff(dist.cdf(k + loc)))
    npt.assert_equal(dist.pmf(k_shifted + loc), 0)