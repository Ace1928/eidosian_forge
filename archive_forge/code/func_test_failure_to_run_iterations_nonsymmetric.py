import itertools
import platform
import sys
import pytest
import numpy as np
from numpy import ones, r_, diag
from numpy.testing import (assert_almost_equal, assert_equal,
from scipy import sparse
from scipy.linalg import eig, eigh, toeplitz, orth
from scipy.sparse import spdiags, diags, eye, csr_matrix
from scipy.sparse.linalg import eigs, LinearOperator
from scipy.sparse.linalg._eigen.lobpcg import lobpcg
from scipy.sparse.linalg._eigen.lobpcg.lobpcg import _b_orthonormalize
from scipy._lib._util import np_long, np_ulong
def test_failure_to_run_iterations_nonsymmetric():
    """Check that the code exists gracefully without breaking
    if the matrix in not symmetric.
    """
    A = np.zeros((10, 10))
    A[0, 1] = 1
    Q = np.ones((10, 1))
    msg = 'Exited at iteration 2|Exited postprocessing with accuracies.*'
    with pytest.warns(UserWarning, match=msg):
        eigenvalues, _ = lobpcg(A, Q, maxiter=20)
    assert np.max(eigenvalues) > 0