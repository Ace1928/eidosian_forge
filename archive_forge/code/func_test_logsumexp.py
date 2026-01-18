import numpy as np
from numpy.testing import (assert_almost_equal, assert_equal, assert_allclose,
from scipy.special import logsumexp, softmax
def test_logsumexp():
    a = np.arange(200)
    desired = np.log(np.sum(np.exp(a)))
    assert_almost_equal(logsumexp(a), desired)
    b = [1000, 1000]
    desired = 1000.0 + np.log(2.0)
    assert_almost_equal(logsumexp(b), desired)
    n = 1000
    b = np.full(n, 10000, dtype='float64')
    desired = 10000.0 + np.log(n)
    assert_almost_equal(logsumexp(b), desired)
    x = np.array([1e-40] * 1000000)
    logx = np.log(x)
    X = np.vstack([x, x])
    logX = np.vstack([logx, logx])
    assert_array_almost_equal(np.exp(logsumexp(logX)), X.sum())
    assert_array_almost_equal(np.exp(logsumexp(logX, axis=0)), X.sum(axis=0))
    assert_array_almost_equal(np.exp(logsumexp(logX, axis=1)), X.sum(axis=1))
    assert_equal(logsumexp(np.inf), np.inf)
    assert_equal(logsumexp(-np.inf), -np.inf)
    assert_equal(logsumexp(np.nan), np.nan)
    assert_equal(logsumexp([-np.inf, -np.inf]), -np.inf)
    assert_array_almost_equal(logsumexp([[10000000000.0, 1e-10], [-10000000000.0, -np.inf]], axis=-1), [10000000000.0, -10000000000.0])
    assert_array_almost_equal(logsumexp([[10000000000.0, 1e-10], [-10000000000.0, -np.inf]], axis=-1, keepdims=True), [[10000000000.0], [-10000000000.0]])
    assert_array_almost_equal(logsumexp([[10000000000.0, 1e-10], [-10000000000.0, -np.inf]], axis=(-1, -2)), 10000000000.0)