from itertools import product
import numpy as np
from numpy.testing import assert_array_equal, assert_equal
import pytest
from scipy.sparse import csr_matrix, coo_matrix, diags
from scipy.sparse.csgraph import (
@pytest.mark.parametrize('num_rows,num_cols', [(0, 0), (2, 0), (0, 3)])
def test_min_weight_full_matching_trivial_graph(num_rows, num_cols):
    biadjacency_matrix = csr_matrix((num_cols, num_rows))
    row_ind, col_ind = min_weight_full_bipartite_matching(biadjacency_matrix)
    assert len(row_ind) == 0
    assert len(col_ind) == 0