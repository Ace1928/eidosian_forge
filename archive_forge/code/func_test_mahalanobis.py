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
def test_mahalanobis(self):
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([1.0, 1.0, 5.0])
    vi = np.array([[2.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 2.0]])
    for x, y in self.cases:
        dist = mahalanobis(x, y, vi)
        assert_almost_equal(dist, np.sqrt(6.0))