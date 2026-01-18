import numpy as np
from numpy.testing import assert_allclose, assert_equal
from scipy import stats
import statsmodels.distributions.tools as dt
def test_bernstein_2d():
    k = 5
    xg1 = np.arange(k) / (k - 1)
    cd2d = xg1[:, None] * xg1
    for evalbp in (dt._eval_bernstein_2d, dt._eval_bernstein_dd):
        k_x = 2 * k
        x2d = np.column_stack(np.unravel_index(np.arange(k_x * k_x), (k_x, k_x))).astype(float)
        x2d /= x2d.max(0)
        res_bp = evalbp(x2d, cd2d)
        assert_allclose(res_bp, np.prod(x2d, axis=1), atol=1e-12)
        x2d = np.column_stack((np.arange(k_x) / (k_x - 1), np.ones(k_x)))
        res_bp = evalbp(x2d, cd2d)
        assert_allclose(res_bp, x2d[:, 0], atol=1e-12)
        x2d = np.column_stack((np.ones(k_x), np.arange(k_x) / (k_x - 1)))
        res_bp = evalbp(x2d, cd2d)
        assert_allclose(res_bp, x2d[:, 1], atol=1e-12)