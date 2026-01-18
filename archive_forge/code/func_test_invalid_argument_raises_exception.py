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
@pytest.mark.parametrize('uplo,trans,diag', [['U', 'N', 'Invalid'], ['U', 'Invalid', 'N'], ['Invalid', 'N', 'N']])
def test_invalid_argument_raises_exception(self, uplo, trans, diag):
    """Test if invalid values of uplo, trans and diag raise exceptions"""
    tbtrs = get_lapack_funcs('tbtrs', dtype=np.float64)
    ab = rand(4, 2)
    b = rand(2, 4)
    assert_raises(Exception, tbtrs, ab, b, uplo, trans, diag)