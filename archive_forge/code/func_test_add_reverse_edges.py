import numpy as np
from numpy.testing import assert_array_equal
import pytest
from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse.csgraph import maximum_flow
from scipy.sparse.csgraph._flow import (
@pytest.mark.parametrize('a,b_data_expected', [([[]], []), ([[0], [0]], []), ([[1, 0, 2], [0, 0, 0], [0, 3, 0]], [1, 2, 0, 0, 3]), ([[9, 8, 7], [4, 5, 6], [0, 0, 0]], [9, 8, 7, 4, 5, 6, 0, 0])])
def test_add_reverse_edges(a, b_data_expected):
    """Test that the reversal of the edges of the input graph works
    as expected.
    """
    a = csr_matrix(a, dtype=np.int32, shape=(len(a), len(a)))
    b = _add_reverse_edges(a)
    assert_array_equal(b.data, b_data_expected)