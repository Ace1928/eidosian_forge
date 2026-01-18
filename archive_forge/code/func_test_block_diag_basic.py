import numpy as np
from numpy import array
from numpy.testing import (assert_equal, assert_,
import pytest
from pytest import raises as assert_raises
from scipy._lib._testutils import check_free_memory
from scipy._lib._util import check_random_state
from scipy.sparse import (csr_matrix, coo_matrix,
from scipy.sparse._construct import rand as sprand
def test_block_diag_basic(self):
    """ basic test for block_diag """
    A = coo_array([[1, 2], [3, 4]])
    B = coo_array([[5], [6]])
    C = coo_array([[7]])
    expected = array([[1, 2, 0, 0], [3, 4, 0, 0], [0, 0, 5, 0], [0, 0, 6, 0], [0, 0, 0, 7]])
    assert_equal(construct.block_diag((A, B, C)).toarray(), expected)