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
def test_eigvalsh_new_args(self):
    a = _random_hermitian_matrix(5)
    w = eigvalsh(a, subset_by_index=[1, 2])
    assert_equal(len(w), 2)
    w2 = eigvalsh(a, subset_by_index=[1, 2])
    assert_equal(len(w2), 2)
    assert_allclose(w, w2)
    b = np.diag([1, 1.2, 1.3, 1.5, 2])
    w3 = eigvalsh(b, subset_by_value=[1, 1.4])
    assert_equal(len(w3), 2)
    assert_allclose(w3, np.array([1.2, 1.3]))