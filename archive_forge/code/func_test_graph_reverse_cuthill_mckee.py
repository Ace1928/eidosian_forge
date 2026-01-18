import numpy as np
from numpy.testing import assert_equal
from scipy.sparse.csgraph import reverse_cuthill_mckee, structural_rank
from scipy.sparse import csc_matrix, csr_matrix, coo_matrix
def test_graph_reverse_cuthill_mckee():
    A = np.array([[1, 0, 0, 0, 1, 0, 0, 0], [0, 1, 1, 0, 0, 1, 0, 1], [0, 1, 1, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0, 1, 0], [1, 0, 1, 0, 1, 0, 0, 0], [0, 1, 0, 0, 0, 1, 0, 1], [0, 0, 0, 1, 0, 0, 1, 0], [0, 1, 0, 0, 0, 1, 0, 1]], dtype=int)
    graph = csr_matrix(A)
    perm = reverse_cuthill_mckee(graph)
    correct_perm = np.array([6, 3, 7, 5, 1, 2, 4, 0])
    assert_equal(perm, correct_perm)
    graph.indices = graph.indices.astype('int64')
    graph.indptr = graph.indptr.astype('int64')
    perm = reverse_cuthill_mckee(graph, True)
    assert_equal(perm, correct_perm)