import numpy as np
from numpy.testing import assert_array_almost_equal, assert_
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix
import pytest
@pytest.mark.parametrize('ax', (-2, -1, 0, 1, None))
def test_argmax_overflow(ax):
    dim = (100000, 100000)
    A = lil_matrix(dim)
    A[-2, -2] = 42
    A[-3, -3] = 0.1234
    A = csc_matrix(A)
    idx = A.argmax(axis=ax)
    if ax is None:
        ii = idx % dim[0]
        jj = idx // dim[0]
    else:
        assert np.count_nonzero(idx) == A.nnz
        ii, jj = (np.max(idx), np.argmax(idx))
    assert A[ii, jj] == A[-2, -2]