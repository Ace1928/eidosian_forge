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
def test_modifies_input(metric):
    X1 = np.asarray([[1.0, 2.0, 3.0], [1.2, 2.3, 3.4], [2.2, 2.3, 4.4], [22.2, 23.3, 44.4]])
    X1_copy = X1.copy()
    cdist(X1, X1, metric)
    pdist(X1, metric)
    assert_array_equal(X1, X1_copy)