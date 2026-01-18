import threading
import itertools
import numpy as np
from numpy.testing import assert_allclose, assert_equal, suppress_warnings
from pytest import raises as assert_raises
import pytest
from numpy import dot, conj, random
from scipy.linalg import eig, eigh
from scipy.sparse import csc_matrix, csr_matrix, diags, rand
from scipy.sparse.linalg import LinearOperator, aslinearoperator
from scipy.sparse.linalg._eigen.arpack import (eigs, eigsh, arpack,
from scipy._lib._gcutils import assert_deallocated, IS_PYPY
def generate_matrix_symmetric(N, pos_definite=False, sparse=False):
    M = np.random.random((N, N))
    M = 0.5 * (M + M.T)
    if pos_definite:
        Id = N * np.eye(N)
        if sparse:
            M = csr_matrix(M)
        M += Id
    elif sparse:
        M = csr_matrix(M)
    return M