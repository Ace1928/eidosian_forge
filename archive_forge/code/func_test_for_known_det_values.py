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
def test_for_known_det_values(self):
    a = np.array([[1, 1, 1, 1, 1, 1, 1, 1], [1, -1, 1, -1, 1, -1, 1, -1], [1, 1, -1, -1, 1, 1, -1, -1], [1, -1, -1, 1, 1, -1, -1, 1], [1, 1, 1, 1, -1, -1, -1, -1], [1, -1, 1, -1, -1, 1, -1, 1], [1, 1, -1, -1, -1, -1, 1, 1], [1, -1, -1, 1, -1, 1, 1, -1]])
    assert_allclose(det(a), 4096.0)
    assert_allclose(det(np.arange(25).reshape(5, 5)), 0.0)
    a = np.array([[0.0 + 0j, 0.0 + 0j, 0.0 - 1j, 1.0 - 1j], [0.0 + 0j, 0.0 + 0j, 1.0 + 0j, 0.0 - 1j], [0.0 + 1j, 1.0 + 1j, 0.0 + 0j, 0.0 + 0j], [1.0 + 0j, 0.0 + 1j, 0.0 + 0j, 0.0 + 0j]], dtype=np.complex64)
    assert_allclose(det(a), 5.0 + 0j)
    a = np.array([[-2.0, -3.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, -4.0, 0.0, -5.0, 1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, -6.0, 0.0, -7.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, -8.0, 0.0, -9.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]]) * 1j
    assert_allclose(det(a), 9.0)