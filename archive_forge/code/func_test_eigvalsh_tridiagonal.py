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
def test_eigvalsh_tridiagonal(self):
    """Compare eigenvalues of eigvalsh_tridiagonal with those of eig."""
    for driver in ('sterf', 'stev', 'stebz', 'stemr', 'auto'):
        w = eigvalsh_tridiagonal(self.d, self.e, lapack_driver=driver)
        assert_array_almost_equal(sort(w), self.w)
    for driver in ('sterf', 'stev'):
        assert_raises(ValueError, eigvalsh_tridiagonal, self.d, self.e, lapack_driver='stev', select='i', select_range=(0, 1))
    for driver in ('stebz', 'stemr', 'auto'):
        w_ind = eigvalsh_tridiagonal(self.d, self.e, select='i', select_range=(0, len(self.d) - 1), lapack_driver=driver)
        assert_array_almost_equal(sort(w_ind), self.w)
        ind1 = 2
        ind2 = 6
        w_ind = eigvalsh_tridiagonal(self.d, self.e, select='i', select_range=(ind1, ind2), lapack_driver=driver)
        assert_array_almost_equal(sort(w_ind), self.w[ind1:ind2 + 1])
        v_lower = self.w[ind1] - 1e-05
        v_upper = self.w[ind2] + 1e-05
        w_val = eigvalsh_tridiagonal(self.d, self.e, select='v', select_range=(v_lower, v_upper), lapack_driver=driver)
        assert_array_almost_equal(sort(w_val), self.w[ind1:ind2 + 1])