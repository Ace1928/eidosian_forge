import numpy as np
from numpy.testing import assert_equal, assert_array_almost_equal
from scipy.sparse import csgraph, csr_array
def test_int64_indices_directed():
    g = csr_array(([1], np.array([[0], [1]], dtype=np.int64)), shape=(2, 2))
    assert g.indices.dtype == np.int64
    n, labels = csgraph.connected_components(g, directed=True, connection='strong')
    assert n == 2
    assert_array_almost_equal(labels, [1, 0])