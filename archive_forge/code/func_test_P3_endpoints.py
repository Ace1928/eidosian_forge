import pytest
import networkx as nx
def test_P3_endpoints(self):
    """Betweenness centrality: P3 endpoints"""
    G = nx.path_graph(3)
    b_answer = {0: 2.0, 1: 3.0, 2: 2.0}
    b = nx.betweenness_centrality(G, weight=None, normalized=False, endpoints=True)
    for n in sorted(G):
        assert b[n] == pytest.approx(b_answer[n], abs=1e-07)
    b_answer = {0: 2 / 3, 1: 1.0, 2: 2 / 3}
    b = nx.betweenness_centrality(G, weight=None, normalized=True, endpoints=True)
    for n in sorted(G):
        assert b[n] == pytest.approx(b_answer[n], abs=1e-07)