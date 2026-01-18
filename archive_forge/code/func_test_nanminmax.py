import contextlib
import functools
import operator
import platform
import itertools
import sys
from scipy._lib import _pep440
import numpy as np
from numpy import (arange, zeros, array, dot, asarray,
import random
from numpy.testing import (assert_equal, assert_array_equal,
from pytest import raises as assert_raises
import scipy.linalg
import scipy.sparse as sparse
from scipy.sparse import (csc_matrix, csr_matrix, dok_matrix,
from scipy.sparse._sputils import (supported_dtypes, isscalarlike,
from scipy.sparse.linalg import splu, expm, inv
from scipy._lib.decorator import decorator
from scipy._lib._util import ComplexWarning
import pytest
def test_nanminmax(self):
    D = matrix(np.arange(50).reshape(5, 10), dtype=float)
    D[1, :] = 0
    D[:, 9] = 0
    D[3, 3] = 0
    D[2, 2] = -1
    D[4, 2] = np.nan
    D[1, 4] = np.nan
    X = self.spcreator(D)
    X_nan_maximum = X.nanmax()
    assert np.isscalar(X_nan_maximum)
    assert X_nan_maximum == np.nanmax(D)
    X_nan_minimum = X.nanmin()
    assert np.isscalar(X_nan_minimum)
    assert X_nan_minimum == np.nanmin(D)
    axes = [-2, -1, 0, 1]
    for axis in axes:
        X_nan_maxima = X.nanmax(axis=axis)
        assert isinstance(X_nan_maxima, coo_matrix)
        assert_allclose(X_nan_maxima.toarray(), np.nanmax(D, axis=axis))
        X_nan_minima = X.nanmin(axis=axis)
        assert isinstance(X_nan_minima, coo_matrix)
        assert_allclose(X_nan_minima.toarray(), np.nanmin(D, axis=axis))