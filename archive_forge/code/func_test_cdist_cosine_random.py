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
def test_cdist_cosine_random(self):
    eps = 1e-14
    X1 = eo['cdist-X1']
    X2 = eo['cdist-X2']
    Y1 = wcdist(X1, X2, 'cosine')

    def norms(X):
        return np.linalg.norm(X, axis=1).reshape(-1, 1)
    Y2 = 1 - np.dot(X1 / norms(X1), (X2 / norms(X2)).T)
    assert_allclose(Y1, Y2, rtol=eps, verbose=verbose > 2)