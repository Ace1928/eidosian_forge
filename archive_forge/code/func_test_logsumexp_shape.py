import numpy as np
from numpy.testing import (assert_almost_equal, assert_equal, assert_allclose,
from scipy.special import logsumexp, softmax
def test_logsumexp_shape():
    a = np.ones((1, 2, 3, 4))
    b = np.ones_like(a)
    r = logsumexp(a, axis=2, b=b)
    assert_equal(r.shape, (1, 2, 4))
    r = logsumexp(a, axis=(1, 3), b=b)
    assert_equal(r.shape, (1, 3))