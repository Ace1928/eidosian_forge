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
@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('shape', [(3, 7), (7, 3), (2 ** 18, 2 ** 18)])
def test_geqrfp_lwork(dtype, shape):
    geqrfp_lwork = get_lapack_funcs('geqrfp_lwork', dtype=dtype)
    m, n = shape
    lwork, info = geqrfp_lwork(m=m, n=n)
    assert_equal(info, 0)