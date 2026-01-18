import pytest
import networkx as nx
from networkx.algorithms import bipartite
from networkx.utils import edges_equal
def test_biadjacency_matrix_empty_graph(self):
    G = nx.empty_graph(2)
    M = nx.bipartite.biadjacency_matrix(G, [0])
    assert np.array_equal(M.toarray(), np.array([[0]]))