import pytest
import networkx as nx
def test_disconnected_path(self):
    """Betweenness centrality: disconnected path"""
    G = nx.Graph()
    nx.add_path(G, [0, 1, 2])
    nx.add_path(G, [3, 4, 5, 6])
    b_answer = {0: 0, 1: 1, 2: 0, 3: 0, 4: 2, 5: 2, 6: 0}
    b = nx.betweenness_centrality(G, weight=None, normalized=False)
    for n in sorted(G):
        assert b[n] == pytest.approx(b_answer[n], abs=1e-07)