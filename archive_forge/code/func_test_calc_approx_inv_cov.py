import numpy as np
from numpy.testing import assert_equal, assert_
from statsmodels.stats.regularized_covariance import (
def test_calc_approx_inv_cov():
    np.random.seed(435265)
    X = np.random.normal(size=(50, 3))
    ghat_l = []
    that_l = []
    for i in range(3):
        ghat = _calc_nodewise_row(X, i, 0.01)
        that = _calc_nodewise_weight(X, ghat, i, 0.01)
        ghat_l.append(ghat)
        that_l.append(that)
    theta_hat = _calc_approx_inv_cov(np.array(ghat_l), np.array(that_l))
    assert_equal(theta_hat.shape, (3, 3))