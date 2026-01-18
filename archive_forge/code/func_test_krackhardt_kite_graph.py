import pytest
import networkx as nx
def test_krackhardt_kite_graph(self):
    """Weighted betweenness centrality: Krackhardt kite graph"""
    G = nx.krackhardt_kite_graph()
    b_answer = {0: 1.667, 1: 1.667, 2: 0.0, 3: 7.333, 4: 0.0, 5: 16.667, 6: 16.667, 7: 28.0, 8: 16.0, 9: 0.0}
    for b in b_answer:
        b_answer[b] /= 2
    b = nx.betweenness_centrality(G, weight='weight', normalized=False)
    for n in sorted(G):
        assert b[n] == pytest.approx(b_answer[n], abs=0.001)