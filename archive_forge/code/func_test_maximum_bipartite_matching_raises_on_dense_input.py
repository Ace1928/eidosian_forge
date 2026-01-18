from itertools import product
import numpy as np
from numpy.testing import assert_array_equal, assert_equal
import pytest
from scipy.sparse import csr_matrix, coo_matrix, diags
from scipy.sparse.csgraph import (
def test_maximum_bipartite_matching_raises_on_dense_input():
    with pytest.raises(TypeError):
        graph = np.array([[0, 1], [0, 0]])
        maximum_bipartite_matching(graph)