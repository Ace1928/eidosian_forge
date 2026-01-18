import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from scipy.sparse import csr_array
from scipy.sparse.csgraph import (breadth_first_tree, depth_first_tree,
@pytest.mark.parametrize('directed', [True, False])
@pytest.mark.parametrize('tree_func', [breadth_first_tree, depth_first_tree])
def test_int64_indices(tree_func, directed):
    g = csr_array(([1], np.array([[0], [1]], dtype=np.int64)), shape=(2, 2))
    assert g.indices.dtype == np.int64
    tree = tree_func(g, 0, directed=directed)
    assert_array_almost_equal(csgraph_to_dense(tree), [[0, 1], [0, 0]])