from itertools import permutations
import pytest
import networkx as nx
def test_degree_p4_weighted(self):
    G = nx.path_graph(4)
    G[1][2]['weight'] = 4
    answer = {1: 2.0, 2: 1.8}
    nd = nx.average_degree_connectivity(G, weight='weight')
    assert nd == answer
    answer = {1: 2.0, 2: 1.5}
    nd = nx.average_degree_connectivity(G)
    assert nd == answer
    D = G.to_directed()
    answer = {2: 2.0, 4: 1.8}
    nd = nx.average_degree_connectivity(D, weight='weight')
    assert nd == answer
    answer = {1: 2.0, 2: 1.8}
    D = G.to_directed()
    nd = nx.average_degree_connectivity(D, weight='weight', source='in', target='in')
    assert nd == answer
    D = G.to_directed()
    nd = nx.average_degree_connectivity(D, source='in', target='out', weight='weight')
    assert nd == answer