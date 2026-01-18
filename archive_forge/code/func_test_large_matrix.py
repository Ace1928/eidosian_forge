import itertools
import platform
import sys
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal,
import pytest
from pytest import raises as assert_raises
from scipy.linalg import (eig, eigvals, lu, svd, svdvals, cholesky, qr,
from scipy.linalg.lapack import (dgbtrf, dgbtrs, zgbtrf, zgbtrs, dsbev,
from scipy.linalg._misc import norm
from scipy.linalg._decomp_qz import _select_function
from scipy.stats import ortho_group
from numpy import (array, diag, full, linalg, argsort, zeros, arange,
from scipy.linalg._testutils import assert_no_overwrite
from scipy.sparse._sputils import matrix
from scipy._lib._testutils import check_free_memory
from scipy.linalg.blas import HAS_ILP64
@pytest.mark.skipif(not HAS_ILP64, reason='64-bit LAPACK required')
@pytest.mark.slow
def test_large_matrix(self):
    check_free_memory(free_mb=17000)
    A = np.zeros([1, 2 ** 31], dtype=np.float32)
    A[0, -1] = 1
    u, s, vh = svd(A, full_matrices=False)
    assert_allclose(s[0], 1.0)
    assert_allclose(u[0, 0] * vh[0, -1], 1.0)