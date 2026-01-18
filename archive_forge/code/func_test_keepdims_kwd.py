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
def test_keepdims_kwd(self):
    a = np.arange(120, dtype='d').reshape(2, 3, 4, 5)
    b = norm(a, ord=np.inf, axis=(1, 0), keepdims=True)
    c = norm(a, ord=1, axis=(0, 1), keepdims=True)
    assert_allclose(b, c)
    assert_(b.shape == c.shape)