import pytest
import networkx as nx
def test_bipartite_k5(self):
    G = nx.complete_bipartite_graph(5, 5)
    assert list(nx.square_clustering(G).values()) == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]