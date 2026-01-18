import pytest
import networkx as nx
def test_K5_endpoints(self):
    """Betweenness centrality: K5 endpoints"""
    G = nx.complete_graph(5)
    b = nx.betweenness_centrality(G, weight=None, normalized=False, endpoints=True)
    b_answer = {0: 4.0, 1: 4.0, 2: 4.0, 3: 4.0, 4: 4.0}
    for n in sorted(G):
        assert b[n] == pytest.approx(b_answer[n], abs=1e-07)
    b = nx.betweenness_centrality(G, weight=None, normalized=True, endpoints=True)
    b_answer = {0: 0.4, 1: 0.4, 2: 0.4, 3: 0.4, 4: 0.4}
    for n in sorted(G):
        assert b[n] == pytest.approx(b_answer[n], abs=1e-07)