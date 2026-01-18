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
def test_call_with_cast_to_complex_without_umfpack(self):
    use_solver(useUmfpack=False)
    solve = factorized(self.A)
    b = random.rand(4)
    for t in [np.complex64, np.complex128]:
        with assert_raises(TypeError, match='Cannot cast array data'):
            solve(b.astype(t))