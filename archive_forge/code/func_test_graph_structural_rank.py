import numpy as np
from numpy.testing import assert_equal
from scipy.sparse.csgraph import reverse_cuthill_mckee, structural_rank
from scipy.sparse import csc_matrix, csr_matrix, coo_matrix
def test_graph_structural_rank():
    A = csc_matrix([[1, 1, 0], [1, 0, 1], [0, 1, 0]])
    assert_equal(structural_rank(A), 3)
    rows = np.array([0, 0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7])
    cols = np.array([0, 1, 2, 3, 4, 2, 5, 2, 6, 0, 1, 3, 5, 6, 7, 4, 5, 5, 6, 2, 6, 2, 4])
    data = np.ones_like(rows)
    B = coo_matrix((data, (rows, cols)), shape=(8, 8))
    assert_equal(structural_rank(B), 6)
    C = csc_matrix([[1, 0, 2, 0], [2, 0, 4, 0]])
    assert_equal(structural_rank(C), 2)
    assert_equal(structural_rank(C.T), 2)