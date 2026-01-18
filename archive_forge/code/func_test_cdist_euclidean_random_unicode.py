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
def test_cdist_euclidean_random_unicode(self):
    eps = 1e-15
    X1 = eo['cdist-X1']
    X2 = eo['cdist-X2']
    Y1 = wcdist_no_const(X1, X2, 'euclidean')
    Y2 = wcdist_no_const(X1, X2, 'test_euclidean')
    assert_allclose(Y1, Y2, rtol=eps, verbose=verbose > 2)