import pytest
import networkx as nx
def test_k_is_3():
    G = nx.Graph([(1, 6), (1, 7), (1, 8), (1, 9), (2, 6), (2, 7), (2, 8), (2, 10), (3, 6), (3, 8), (3, 9), (3, 10), (4, 7), (4, 8), (4, 9), (4, 10), (5, 6), (5, 7), (5, 9), (5, 10)])
    assert nx.bipartite.maximal_extendability(G) == 3