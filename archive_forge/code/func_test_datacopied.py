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
def test_datacopied(self):
    from scipy.linalg._decomp import _datacopied
    M = matrix([[0, 1], [2, 3]])
    A = asarray(M)
    L = M.tolist()
    M2 = M.copy()

    class Fake1:

        def __array__(self):
            return A

    class Fake2:
        __array_interface__ = A.__array_interface__
    F1 = Fake1()
    F2 = Fake2()
    for item, status in [(M, False), (A, False), (L, True), (M2, False), (F1, False), (F2, False)]:
        arr = asarray(item)
        assert_equal(_datacopied(arr, item), status, err_msg=repr(item))