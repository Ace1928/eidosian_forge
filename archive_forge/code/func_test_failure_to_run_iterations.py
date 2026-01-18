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
@pytest.mark.filterwarnings('ignore:Failed at iteration')
@pytest.mark.filterwarnings('ignore:Exited at iteration')
@pytest.mark.filterwarnings('ignore:Exited postprocessing')
def test_failure_to_run_iterations():
    """Check that the code exits gracefully without breaking. Issue #10974.
    The code may or not issue a warning, filtered out. Issue #15935, #17954.
    """
    rnd = np.random.RandomState(0)
    X = rnd.standard_normal((100, 10))
    A = X @ X.T
    Q = rnd.standard_normal((X.shape[0], 4))
    eigenvalues, _ = lobpcg(A, Q, maxiter=40, tol=1e-12)
    assert np.max(eigenvalues) > 0