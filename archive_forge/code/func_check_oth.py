import numpy.testing as npt
from numpy.testing import assert_allclose
import numpy as np
import pytest
from scipy import stats
from .common_tests import (check_normalization, check_moment,
from scipy.stats._distr_params import distdiscrete, invdistdiscrete
from scipy.stats._distn_infrastructure import rv_discrete_frozen
def check_oth(distfn, arg, supp, msg):
    npt.assert_allclose(distfn.sf(supp, *arg), 1.0 - distfn.cdf(supp, *arg), atol=1e-10, rtol=1e-10)
    q = np.linspace(0.01, 0.99, 20)
    npt.assert_allclose(distfn.isf(q, *arg), distfn.ppf(1.0 - q, *arg), atol=1e-10, rtol=1e-10)
    median_sf = distfn.isf(0.5, *arg)
    npt.assert_(distfn.sf(median_sf - 1, *arg) > 0.5)
    npt.assert_(distfn.cdf(median_sf + 1, *arg) > 0.5)