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
@pytest.mark.filterwarnings('ignore:Exited at iteration')
@pytest.mark.filterwarnings('ignore:Exited postprocessing')
def test_inplace_warning():
    """Check lobpcg gives a warning in '_b_orthonormalize'
    that in-place orthogonalization is impossible due to dtype mismatch.
    """
    rnd = np.random.RandomState(0)
    n = 6
    m = 1
    vals = -np.arange(1, n + 1)
    A = diags([vals], [0], (n, n))
    A = A.astype(np.cdouble)
    X = rnd.standard_normal((n, m))
    with pytest.warns(UserWarning, match='Inplace update'):
        eigvals, _ = lobpcg(A, X, maxiter=2, verbosityLevel=1)