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
def test_braycurtis():
    assert_almost_equal(wbraycurtis([1, 2, 3], [2, 4, 6]), 1.0 / 3, decimal=15)
    assert_almost_equal(wbraycurtis([1, 1, 0, 0], [1, 0, 1, 0]), 0.5, decimal=15)