import pytest
import networkx as nx
from networkx.algorithms import approximation as approx
def test_global_node_connectivity():
    G = nx.Graph()
    G.add_edges_from([(1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (2, 6), (3, 4), (3, 6), (4, 6), (4, 7), (5, 7), (6, 8), (6, 9), (7, 8), (7, 10), (8, 11), (9, 10), (9, 11), (10, 11)])
    assert 2 == approx.local_node_connectivity(G, 1, 11)
    assert 2 == approx.node_connectivity(G)
    assert 2 == approx.node_connectivity(G, 1, 11)