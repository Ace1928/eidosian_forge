import pytest
import networkx as nx
from networkx.algorithms.isomorphism.isomorph import graph_could_be_isomorphic
def test__LCF_graph(self):
    G = nx.LCF_graph(-10, [1, 2], 100)
    assert is_isomorphic(G, null)
    G = nx.LCF_graph(0, [1, 2], 3)
    assert is_isomorphic(G, null)
    G = nx.LCF_graph(0, [1, 2], 10)
    assert is_isomorphic(G, null)
    for a, b, c in [(5, [], 0), (10, [], 0), (5, [], 1), (10, [], 10)]:
        G = nx.LCF_graph(a, b, c)
        assert is_isomorphic(G, nx.cycle_graph(a))
    G = nx.LCF_graph(6, [3, -3], 3)
    utility_graph = nx.complete_bipartite_graph(3, 3)
    assert is_isomorphic(G, utility_graph)