import numpy as np
from numpy.testing import assert_equal, assert_
from statsmodels.stats.regularized_covariance import (
def test_calc_nodewise_row():
    np.random.seed(435265)
    X = np.random.normal(size=(50, 3))
    ghat = _calc_nodewise_row(X, 0, 0.01)
    assert_equal(ghat.shape, (2,))