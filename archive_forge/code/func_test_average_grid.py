import numpy as np
from numpy.testing import assert_allclose, assert_equal
from scipy import stats
import statsmodels.distributions.tools as dt
def test_average_grid():
    x1 = np.arange(1, 4)
    x2 = np.arange(4)
    y = x1[:, None] * x2
    res1 = np.array([[0.75, 2.25, 3.75], [1.25, 3.75, 6.25]])
    res0 = dt.average_grid(y, coords=[x1, x2])
    assert_allclose(res0, res1, rtol=1e-13)
    res0 = dt.average_grid(y, coords=[x1, x2], _method='slicing')
    assert_allclose(res0, res1, rtol=1e-13)
    res0 = dt.average_grid(y, coords=[x1, x2], _method='convolve')
    assert_allclose(res0, res1, rtol=1e-13)
    res0 = dt.average_grid(y, coords=[x1 / x1.max(), x2 / x2.max()])
    assert_allclose(res0, res1 / x1.max() / x2.max(), rtol=1e-13)
    res0 = dt.average_grid(y, coords=[x1 / x1.max(), x2 / x2.max()], _method='convolve')
    assert_allclose(res0, res1 / x1.max() / x2.max(), rtol=1e-13)