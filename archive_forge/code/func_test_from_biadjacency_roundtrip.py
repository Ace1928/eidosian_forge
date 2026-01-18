import pytest
import networkx as nx
from networkx.algorithms import bipartite
from networkx.utils import edges_equal
def test_from_biadjacency_roundtrip(self):
    B1 = nx.path_graph(5)
    M = bipartite.biadjacency_matrix(B1, [0, 2, 4])
    B2 = bipartite.from_biadjacency_matrix(M)
    assert nx.is_isomorphic(B1, B2)