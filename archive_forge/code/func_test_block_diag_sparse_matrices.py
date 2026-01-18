import numpy as np
from numpy import array
from numpy.testing import (assert_equal, assert_,
import pytest
from pytest import raises as assert_raises
from scipy._lib._testutils import check_free_memory
from scipy._lib._util import check_random_state
from scipy.sparse import (csr_matrix, coo_matrix,
from scipy.sparse._construct import rand as sprand
from scipy.sparse._sputils import matrix
def test_block_diag_sparse_matrices(self):
    """ block_diag with sparse matrices """
    sparse_col_matrices = [coo_matrix([[1, 2, 3]], shape=(1, 3)), coo_matrix([[4, 5]], shape=(1, 2))]
    block_sparse_cols_matrices = construct.block_diag(sparse_col_matrices)
    assert_equal(block_sparse_cols_matrices.toarray(), array([[1, 2, 3, 0, 0], [0, 0, 0, 4, 5]]))
    sparse_row_matrices = [coo_matrix([[1], [2], [3]], shape=(3, 1)), coo_matrix([[4], [5]], shape=(2, 1))]
    block_sparse_row_matrices = construct.block_diag(sparse_row_matrices)
    assert_equal(block_sparse_row_matrices.toarray(), array([[1, 0], [2, 0], [3, 0], [0, 4], [0, 5]]))