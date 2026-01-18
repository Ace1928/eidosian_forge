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
def test_gh_3054(self):
    a = [[1]]
    b = [[0]]
    w, vr = eig(a, b, homogeneous_eigvals=True)
    assert_allclose(w[1, 0], 0)
    assert_(w[0, 0] != 0)
    assert_allclose(vr, 1)
    w, vr = eig(a, b)
    assert_equal(w, np.inf)
    assert_allclose(vr, 1)