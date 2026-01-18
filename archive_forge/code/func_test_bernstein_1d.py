import numpy as np
from numpy.testing import assert_allclose, assert_equal
from scipy import stats
import statsmodels.distributions.tools as dt
def test_bernstein_1d():
    k = 5
    xg1 = np.arange(k) / (k - 1)
    xg2 = np.arange(2 * k) / (2 * k - 1)
    res_bp = dt._eval_bernstein_1d(xg2, xg1)
    assert_allclose(res_bp, xg2, atol=1e-12)
    res_bp = dt._eval_bernstein_1d(xg2, xg1, method='beta')
    assert_allclose(res_bp, xg2, atol=1e-12)
    res_bp = dt._eval_bernstein_1d(xg2, xg1, method='bpoly')
    assert_allclose(res_bp, xg2, atol=1e-12)