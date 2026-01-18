import numpy.testing as npt
from numpy.testing import assert_allclose
import numpy as np
import pytest
from scipy import stats
from .common_tests import (check_normalization, check_moment,
from scipy.stats._distr_params import distdiscrete, invdistdiscrete
from scipy.stats._distn_infrastructure import rv_discrete_frozen
@pytest.mark.parametrize('distname, args', invdistdiscrete)
def test_cdf_gh13280_regression(distname, args):
    dist = getattr(stats, distname)
    x = np.arange(-2, 15)
    vals = dist.cdf(x, *args)
    expected = np.nan
    npt.assert_equal(vals, expected)