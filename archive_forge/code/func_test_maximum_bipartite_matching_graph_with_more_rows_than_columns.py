from itertools import product
import numpy as np
from numpy.testing import assert_array_equal, assert_equal
import pytest
from scipy.sparse import csr_matrix, coo_matrix, diags
from scipy.sparse.csgraph import (
def test_maximum_bipartite_matching_graph_with_more_rows_than_columns():
    graph = csr_matrix([[1, 1], [1, 0], [0, 1]])
    x = maximum_bipartite_matching(graph, perm_type='column')
    y = maximum_bipartite_matching(graph, perm_type='row')
    assert_array_equal(np.array([0, -1, 1]), x)
    assert_array_equal(np.array([0, 2]), y)