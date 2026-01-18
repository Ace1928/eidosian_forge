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
@sup_sparse_efficiency
def test_example_comparison(self):
    row = array([0, 0, 1, 2, 2, 2])
    col = array([0, 2, 2, 0, 1, 2])
    data = array([1, 2, 3, -4, 5, 6])
    sM = csr_matrix((data, (row, col)), shape=(3, 3), dtype=float)
    M = sM.toarray()
    row = array([0, 0, 1, 1, 0, 0])
    col = array([0, 2, 1, 1, 0, 0])
    data = array([1, 1, 1, 1, 1, 1])
    sN = csr_matrix((data, (row, col)), shape=(3, 3), dtype=float)
    N = sN.toarray()
    sX = spsolve(sM, sN)
    X = scipy.linalg.solve(M, N)
    assert_array_almost_equal(X, sX.toarray())