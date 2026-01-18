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
@pytest.mark.skipif(not HAS_ILP64, reason='64-bit BLAS required')
def test_large_vector(self):
    check_free_memory(free_mb=17000)
    x = np.zeros([2 ** 31], dtype=np.float64)
    x[-1] = 1
    res = norm(x)
    del x
    assert_allclose(res, 1.0)