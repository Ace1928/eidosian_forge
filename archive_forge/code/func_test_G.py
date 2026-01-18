import pytest
import networkx as nx
def test_G(self):
    """Weighted betweenness centrality: G"""
    G = weighted_G()
    b_answer = {0: 2.0, 1: 0.0, 2: 4.0, 3: 3.0, 4: 4.0, 5: 0.0}
    b = nx.betweenness_centrality(G, weight='weight', normalized=False)
    for n in sorted(G):
        assert b[n] == pytest.approx(b_answer[n], abs=1e-07)