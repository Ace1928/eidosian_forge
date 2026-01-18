import numpy as np
from numpy.testing import (
import pytest
from scipy import stats
from statsmodels.datasets import macrodata
from statsmodels.regression.linear_model import OLS, WLS
import statsmodels.stats.sandwich_covariance as sw
from statsmodels.tools.sm_exceptions import InvalidTestWarning
from statsmodels.tools.tools import add_constant
from .results import (
def test_cov_type_fixed_scale():
    xdata = np.array([0, 1, 2, 3, 4, 5])
    ydata = np.array([1, 1, 5, 7, 8, 12])
    sigma = np.array([1, 2, 1, 2, 1, 2])
    xdata = np.column_stack((xdata, np.ones(len(xdata))))
    weights = 1.0 / sigma ** 2
    res = WLS(ydata, xdata, weights=weights).fit()
    assert_allclose(res.bse, [0.20659803, 0.57204404], rtol=0.001)
    res = WLS(ydata, xdata, weights=weights).fit()
    assert_allclose(res.bse, [0.20659803, 0.57204404], rtol=0.001)
    res = WLS(ydata, xdata, weights=weights).fit(cov_type='fixed scale')
    assert_allclose(res.bse, [0.30714756, 0.85045308], rtol=0.001)
    res = WLS(ydata, xdata, weights=weights / 9.0).fit(cov_type='fixed scale')
    assert_allclose(res.bse, [3 * 0.30714756, 3 * 0.85045308], rtol=0.001)
    res = WLS(ydata, xdata, weights=weights).fit(cov_type='fixed scale', cov_kwds={'scale': 9})
    assert_allclose(res.bse, [3 * 0.30714756, 3 * 0.85045308], rtol=0.001)