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
def random_triangle_matrix(n, lower=True):
    A = scipy.sparse.random(n, n, density=0.1, format='coo')
    if lower:
        A = scipy.sparse.tril(A)
    else:
        A = scipy.sparse.triu(A)
    A = A.tocsr(copy=False)
    for i in range(n):
        A[i, i] = np.random.rand() + 1
    return A