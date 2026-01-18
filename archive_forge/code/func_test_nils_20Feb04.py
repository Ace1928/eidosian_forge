import itertools
import warnings
import numpy as np
from numpy import (arange, array, dot, zeros, identity, conjugate, transpose,
from numpy.random import random
from numpy.testing import (assert_equal, assert_almost_equal, assert_,
import pytest
from pytest import raises as assert_raises
from scipy.linalg import (solve, inv, det, lstsq, pinv, pinvh, norm,
from scipy.linalg._testutils import assert_no_overwrite
from scipy._lib._testutils import check_free_memory, IS_MUSL
from scipy.linalg.blas import HAS_ILP64
from scipy._lib.deprecation import _NoValue
def test_nils_20Feb04(self):
    n = 2
    A = random([n, n]) + random([n, n]) * 1j
    X = zeros((n, n), 'D')
    Ainv = inv(A)
    R = identity(n) + identity(n) * 0j
    for i in arange(0, n):
        r = R[:, i]
        X[:, i] = solve(A, r)
    assert_array_almost_equal(X, Ainv)