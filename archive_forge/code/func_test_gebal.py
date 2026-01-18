import sys
from functools import reduce
from numpy.testing import (assert_equal, assert_array_almost_equal, assert_,
import pytest
from pytest import raises as assert_raises
import numpy as np
from numpy import (eye, ones, zeros, zeros_like, triu, tril, tril_indices,
from numpy.random import rand, randint, seed
from scipy.linalg import (_flapack as flapack, lapack, inv, svd, cholesky,
from scipy.linalg.lapack import _compute_lwork
from scipy.stats import ortho_group, unitary_group
import scipy.sparse as sps
from scipy.linalg.lapack import get_lapack_funcs
from scipy.linalg.blas import get_blas_funcs
def test_gebal(self):
    a = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    a1 = [[1, 0, 0, 0.0003], [4, 0, 0, 0.002], [7, 1, 0, 0], [0, 1, 0, 0]]
    for p in 'sdzc':
        f = getattr(flapack, p + 'gebal', None)
        if f is None:
            continue
        ba, lo, hi, pivscale, info = f(a)
        assert_(not info, repr(info))
        assert_array_almost_equal(ba, a)
        assert_equal((lo, hi), (0, len(a[0]) - 1))
        assert_array_almost_equal(pivscale, np.ones(len(a)))
        ba, lo, hi, pivscale, info = f(a1, permute=1, scale=1)
        assert_(not info, repr(info))