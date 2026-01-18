import pytest
import networkx as nx
def test_degree_k4(self):
    G = nx.complete_graph(4)
    answer = {0: 3, 1: 3, 2: 3, 3: 3}
    nd = nx.average_neighbor_degree(G)
    assert nd == answer
    D = G.to_directed()
    nd = nx.average_neighbor_degree(D)
    assert nd == answer
    D = G.to_directed()
    nd = nx.average_neighbor_degree(D)
    assert nd == answer
    D = G.to_directed()
    nd = nx.average_neighbor_degree(D, source='in', target='in')
    assert nd == answer