import pytest
import networkx as nx
def test_arbitrary_weights(self):
    G = nx.DiGraph()
    G.add_edge(0, 1, weight=1)
    G.add_edge(0, 2, weight=1)
    S = nx.stochastic_graph(G)
    assert sorted(S.edges(data=True)) == [(0, 1, {'weight': 0.5}), (0, 2, {'weight': 0.5})]