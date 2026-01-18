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
def test_correlation(self):
    xm = np.array([-1.0, 0, 1.0])
    ym = np.array([-4.0 / 3, -4.0 / 3, 5.0 - 7.0 / 3])
    for x, y in self.cases:
        dist = wcorrelation(x, y)
        assert_almost_equal(dist, 1.0 - np.dot(xm, ym) / (norm(xm) * norm(ym)))