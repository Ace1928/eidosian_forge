import sys
import threading
import numpy as np
from numpy import array, finfo, arange, eye, all, unique, ones, dot
import numpy.random as random
from numpy.testing import (
import pytest
from pytest import raises as assert_raises
import scipy.linalg
from scipy.linalg import norm, inv
from scipy.sparse import (spdiags, SparseEfficiencyWarning, csc_matrix,
from scipy.sparse.linalg import SuperLU
from scipy.sparse.linalg._dsolve import (spsolve, use_solver, splu, spilu,
import scipy.sparse
from scipy._lib._testutils import check_free_memory
from scipy._lib._util import ComplexWarning
@pytest.mark.skipif(not hasattr(sys, 'getrefcount'), reason='no sys.getrefcount')
def test_lu_refcount(self):
    n = 30
    a = random.random((n, n))
    a[a < 0.95] = 0
    a += 4 * eye(n)
    a_ = csc_matrix(a)
    lu = splu(a_)
    rc = sys.getrefcount(lu)
    for attr in ('perm_r', 'perm_c'):
        perm = getattr(lu, attr)
        assert_equal(sys.getrefcount(lu), rc + 1)
        del perm
        assert_equal(sys.getrefcount(lu), rc)