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
def test_jensenshannon():
    assert_almost_equal(jensenshannon([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 2.0), 1.0)
    assert_almost_equal(jensenshannon([1.0, 0.0], [0.5, 0.5]), 0.46450140402245893)
    assert_almost_equal(jensenshannon([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]), 0.0)
    assert_almost_equal(jensenshannon([[1.0, 2.0]], [[0.5, 1.5]], axis=0), [0.0, 0.0])
    assert_almost_equal(jensenshannon([[1.0, 2.0]], [[0.5, 1.5]], axis=1), [0.0649045])
    assert_almost_equal(jensenshannon([[1.0, 2.0]], [[0.5, 1.5]], axis=0, keepdims=True), [[0.0, 0.0]])
    assert_almost_equal(jensenshannon([[1.0, 2.0]], [[0.5, 1.5]], axis=1, keepdims=True), [[0.0649045]])
    a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    b = np.array([[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]])
    assert_almost_equal(jensenshannon(a, b, axis=0), [0.1954288, 0.1447697, 0.1138377, 0.0927636])
    assert_almost_equal(jensenshannon(a, b, axis=1), [0.1402339, 0.0399106, 0.0201815])