import math
import pytest
import networkx as nx
def test_K5_unweighted(self):
    """Katz centrality: K5"""
    G = nx.complete_graph(5)
    alpha = 0.1
    b = nx.katz_centrality(G, alpha, weight=None)
    v = math.sqrt(1 / 5.0)
    b_answer = dict.fromkeys(G, v)
    for n in sorted(G):
        assert b[n] == pytest.approx(b_answer[n], abs=1e-07)
    b = nx.eigenvector_centrality_numpy(G, weight=None)
    for n in sorted(G):
        assert b[n] == pytest.approx(b_answer[n], abs=0.001)