import sys
import os.path
from functools import wraps, partial
import weakref
import numpy as np
import warnings
from numpy.linalg import norm
from numpy.testing import (verbose, assert_,
import pytest
import scipy.spatial.distance
from scipy.spatial.distance import (
from scipy.spatial.distance import (braycurtis, canberra, chebyshev, cityblock,
from scipy._lib._util import np_long, np_ulong
def test_euclideans():
    x1 = np.array([1, 1, 1])
    x2 = np.array([0, 0, 0])
    assert_almost_equal(wsqeuclidean(x1, x2), 3.0, decimal=14)
    assert_almost_equal(weuclidean(x1, x2), np.sqrt(3), decimal=14)
    with pytest.raises(ValueError, match='Input vector should be 1-D'):
        (weuclidean(x1[np.newaxis, :], x2[np.newaxis, :]), np.sqrt(3))
    with pytest.raises(ValueError, match='Input vector should be 1-D'):
        wsqeuclidean(x1[np.newaxis, :], x2[np.newaxis, :])
    with pytest.raises(ValueError, match='Input vector should be 1-D'):
        wsqeuclidean(x1[:, np.newaxis], x2[:, np.newaxis])
    x = np.arange(4).reshape(2, 2)
    with pytest.raises(ValueError):
        weuclidean(x, x)
    with pytest.raises(ValueError):
        wsqeuclidean(x, x)
    rs = np.random.RandomState(1234567890)
    x = rs.rand(10)
    y = rs.rand(10)
    d1 = weuclidean(x, y)
    d2 = wsqeuclidean(x, y)
    assert_almost_equal(d1 ** 2, d2, decimal=14)