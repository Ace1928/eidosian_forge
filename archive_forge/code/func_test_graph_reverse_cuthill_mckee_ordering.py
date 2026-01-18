import numpy as np
from numpy.testing import assert_equal
from scipy.sparse.csgraph import reverse_cuthill_mckee, structural_rank
from scipy.sparse import csc_matrix, csr_matrix, coo_matrix
def test_graph_reverse_cuthill_mckee_ordering():
    data = np.ones(63, dtype=int)
    rows = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 10, 10, 10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15, 15, 15])
    cols = np.array([0, 2, 5, 8, 10, 1, 3, 9, 11, 0, 2, 7, 10, 1, 3, 11, 4, 6, 12, 14, 0, 7, 13, 15, 4, 6, 14, 2, 5, 7, 15, 0, 8, 10, 13, 1, 9, 11, 0, 2, 8, 10, 15, 1, 3, 9, 11, 4, 12, 14, 5, 8, 13, 15, 4, 6, 12, 14, 5, 7, 10, 13, 15])
    graph = coo_matrix((data, (rows, cols))).tocsr()
    perm = reverse_cuthill_mckee(graph)
    correct_perm = np.array([12, 14, 4, 6, 10, 8, 2, 15, 0, 13, 7, 5, 9, 11, 1, 3])
    assert_equal(perm, correct_perm)