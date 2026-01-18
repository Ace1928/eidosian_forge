import pytest
import networkx as nx
def test_k5_signed(self):
    G = nx.complete_graph(5)
    assert list(nx.clustering(G).values()) == [1, 1, 1, 1, 1]
    assert nx.average_clustering(G) == 1
    G.remove_edge(1, 2)
    G.add_edge(0, 1, weight=-1)
    assert list(nx.clustering(G, weight='weight').values()) == [1 / 6, -1 / 3, 1, 3 / 6, 3 / 6]