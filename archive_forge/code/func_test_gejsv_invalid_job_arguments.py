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
@pytest.mark.parametrize('kwargs', ({'joba': 9}, {'jobu': 9}, {'jobv': 9}, {'jobr': 9}, {'jobt': 9}, {'jobp': 9}))
def test_gejsv_invalid_job_arguments(kwargs):
    """Test invalid job arguments raise an Exception"""
    A = np.ones((2, 2), dtype=float)
    gejsv = get_lapack_funcs('gejsv', dtype=float)
    assert_raises(Exception, gejsv, A, **kwargs)