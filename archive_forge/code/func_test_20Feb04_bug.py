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
def test_20Feb04_bug(self):
    a = [[1, 1], [1.0, 0]]
    x0 = solve(a, [1, 0j])
    assert_array_almost_equal(dot(a, x0), [1, 0])
    a = [[1, 1], [1.2, 0]]
    b = [1, 0j]
    x0 = solve(a, b)
    assert_array_almost_equal(dot(a, x0), [1, 0])