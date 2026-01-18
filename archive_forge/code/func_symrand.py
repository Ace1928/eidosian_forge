import itertools
import platform
import sys
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal,
import pytest
from pytest import raises as assert_raises
from scipy.linalg import (eig, eigvals, lu, svd, svdvals, cholesky, qr,
from scipy.linalg.lapack import (dgbtrf, dgbtrs, zgbtrf, zgbtrs, dsbev,
from scipy.linalg._misc import norm
from scipy.linalg._decomp_qz import _select_function
from scipy.stats import ortho_group
from numpy import (array, diag, full, linalg, argsort, zeros, arange,
from scipy.linalg._testutils import assert_no_overwrite
from scipy.sparse._sputils import matrix
from scipy._lib._testutils import check_free_memory
from scipy.linalg.blas import HAS_ILP64
def symrand(dim_or_eigv, rng):
    """Return a random symmetric (Hermitian) matrix.

    If 'dim_or_eigv' is an integer N, return a NxN matrix, with eigenvalues
        uniformly distributed on (-1,1).

    If 'dim_or_eigv' is  1-D real array 'a', return a matrix whose
                      eigenvalues are 'a'.
    """
    if isinstance(dim_or_eigv, int):
        dim = dim_or_eigv
        d = rng.random(dim) * 2 - 1
    elif isinstance(dim_or_eigv, ndarray) and len(dim_or_eigv.shape) == 1:
        dim = dim_or_eigv.shape[0]
        d = dim_or_eigv
    else:
        raise TypeError('input type not supported.')
    v = ortho_group.rvs(dim)
    h = v.T.conj() @ diag(d) @ v
    h = 0.5 * (h.T + h)
    return h