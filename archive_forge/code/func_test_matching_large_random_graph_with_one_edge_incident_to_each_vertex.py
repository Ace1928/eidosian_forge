from itertools import product
import numpy as np
from numpy.testing import assert_array_equal, assert_equal
import pytest
from scipy.sparse import csr_matrix, coo_matrix, diags
from scipy.sparse.csgraph import (
def test_matching_large_random_graph_with_one_edge_incident_to_each_vertex():
    np.random.seed(42)
    A = diags(np.ones(25), offsets=0, format='csr')
    rand_perm = np.random.permutation(25)
    rand_perm2 = np.random.permutation(25)
    Rrow = np.arange(25)
    Rcol = rand_perm
    Rdata = np.ones(25, dtype=int)
    Rmat = coo_matrix((Rdata, (Rrow, Rcol))).tocsr()
    Crow = rand_perm2
    Ccol = np.arange(25)
    Cdata = np.ones(25, dtype=int)
    Cmat = coo_matrix((Cdata, (Crow, Ccol))).tocsr()
    B = Rmat * A * Cmat
    perm = maximum_bipartite_matching(B, perm_type='row')
    Rrow = np.arange(25)
    Rcol = perm
    Rdata = np.ones(25, dtype=int)
    Rmat = coo_matrix((Rdata, (Rrow, Rcol))).tocsr()
    C1 = Rmat * B
    perm2 = maximum_bipartite_matching(B, perm_type='column')
    Crow = perm2
    Ccol = np.arange(25)
    Cdata = np.ones(25, dtype=int)
    Cmat = coo_matrix((Cdata, (Crow, Ccol))).tocsr()
    C2 = B * Cmat
    assert_equal(any(C1.diagonal() == 0), False)
    assert_equal(any(C2.diagonal() == 0), False)