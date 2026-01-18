import numpy as np
from numpy.testing import assert_array_equal
import pytest
from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse.csgraph import maximum_flow
from scipy.sparse.csgraph._flow import (
@pytest.mark.parametrize('a,expected', [([[]], []), ([[0]], []), ([[1]], [0]), ([[0, 1], [10, 0]], [1, 0]), ([[1, 0, 2], [0, 0, 3], [4, 5, 0]], [0, 3, 4, 1, 2])])
def test_make_edge_pointers(a, expected):
    a = csr_matrix(a, dtype=np.int32)
    rev_edge_ptr = _make_edge_pointers(a)
    assert_array_equal(rev_edge_ptr, expected)