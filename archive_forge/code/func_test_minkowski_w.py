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
def test_minkowski_w():
    arr_in = np.array([[83.33333333, 100.0, 83.33333333, 100.0, 36.0, 60.0, 90.0, 150.0, 24.0, 48.0], [83.33333333, 100.0, 83.33333333, 100.0, 36.0, 60.0, 90.0, 150.0, 24.0, 48.0]])
    p0 = pdist(arr_in, metric='minkowski', p=1, w=None)
    c0 = cdist(arr_in, arr_in, metric='minkowski', p=1, w=None)
    p1 = pdist(arr_in, metric='minkowski', p=1)
    c1 = cdist(arr_in, arr_in, metric='minkowski', p=1)
    assert_allclose(p0, p1, rtol=1e-15)
    assert_allclose(c0, c1, rtol=1e-15)