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
def test_input_types(self):
    A = array([[1.0, 0.0], [1.0, 2.0]])
    b = array([[2.0, 0.0], [2.0, 2.0]])
    for matrix_type in (array, csc_matrix, csr_matrix):
        x = spsolve_triangular(matrix_type(A), b, lower=True)
        assert_array_almost_equal(A.dot(x), b)