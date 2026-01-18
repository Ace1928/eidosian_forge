import sys
import numpy as np
import numpy.testing as npt
import pytest
from pytest import raises as assert_raises
from scipy.integrate import IntegrationWarning
import itertools
from scipy import stats
from .common_tests import (check_normalization, check_moment,
from scipy.stats._distr_params import distcont
from scipy.stats._distn_infrastructure import rv_continuous_frozen
@pytest.mark.parametrize('x,n,sf,cdf,pdf,rtol', [(2e-05, 1000000000, 0.44932297307934443, 0.5506770269206556, 35946.13739499628, 5e-15), (2e-09, 1000000000, 0.9999999906111111, 9.388888844813272e-09, 8.666666585296298, 5e-14), (0.0005, 1000000000, 7.122201943309037e-218, 1.0, 1.4244408634752703e-211, 5e-14)])
def test_gh17775_regression(x, n, sf, cdf, pdf, rtol):
    ks = stats.ksone
    vals = np.array([ks.sf(x, n), ks.cdf(x, n), ks.pdf(x, n)])
    expected = np.array([sf, cdf, pdf])
    npt.assert_allclose(vals, expected, rtol=rtol)
    npt.assert_equal(vals[0] + vals[1], 1.0)
    npt.assert_allclose([ks.isf(sf, n)], [x], rtol=1e-08)