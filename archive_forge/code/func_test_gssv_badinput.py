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
def test_gssv_badinput(self):
    N = 10
    d = arange(N) + 1.0
    A = spdiags((d, 2 * d, d[::-1]), (-3, 0, 5), N, N)
    for spmatrix in (csc_matrix, csr_matrix):
        A = spmatrix(A)
        b = np.arange(N)

        def not_c_contig(x):
            return x.repeat(2)[::2]

        def not_1dim(x):
            return x[:, None]

        def bad_type(x):
            return x.astype(bool)

        def too_short(x):
            return x[:-1]
        badops = [not_c_contig, not_1dim, bad_type, too_short]
        for badop in badops:
            msg = f'{spmatrix!r} {badop!r}'
            assert_raises((ValueError, TypeError), _superlu.gssv, N, A.nnz, badop(A.data), A.indices, A.indptr, b, int(spmatrix == csc_matrix), err_msg=msg)
            assert_raises((ValueError, TypeError), _superlu.gssv, N, A.nnz, A.data, badop(A.indices), A.indptr, b, int(spmatrix == csc_matrix), err_msg=msg)
            assert_raises((ValueError, TypeError), _superlu.gssv, N, A.nnz, A.data, A.indices, badop(A.indptr), b, int(spmatrix == csc_matrix), err_msg=msg)