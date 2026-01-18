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
def test_empty_edge_cases(self):
    assert_allclose(det(np.empty([0, 0])), 1.0)
    assert_allclose(det(np.empty([0, 0, 0])), np.array([]))
    assert_allclose(det(np.empty([3, 0, 0])), np.array([1.0, 1.0, 1.0]))
    with assert_raises(ValueError, match='Last 2 dimensions'):
        det(np.empty([0, 0, 3]))
    with assert_raises(ValueError, match='at least two-dimensional'):
        det(np.array([]))
    with assert_raises(ValueError, match='Last 2 dimensions'):
        det(np.array([[]]))
    with assert_raises(ValueError, match='Last 2 dimensions'):
        det(np.array([[[]]]))