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
def test_non_square(self):
    A = ones((3, 4))
    b = ones((4, 1))
    assert_raises(ValueError, spsolve, A, b)
    A2 = csc_matrix(eye(3))
    b2 = array([1.0, 2.0])
    assert_raises(ValueError, spsolve, A2, b2)