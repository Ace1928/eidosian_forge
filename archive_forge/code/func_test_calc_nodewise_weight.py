import numpy as np
from numpy.testing import assert_equal, assert_
from statsmodels.stats.regularized_covariance import (
def test_calc_nodewise_weight():
    np.random.seed(435265)
    X = np.random.normal(size=(50, 3))
    ghat = np.random.normal(size=2)
    that = _calc_nodewise_weight(X, ghat, 0, 0.01)
    assert_(isinstance(that, float))