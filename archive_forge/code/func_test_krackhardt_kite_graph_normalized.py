import pytest
import networkx as nx
def test_krackhardt_kite_graph_normalized(self):
    """Weighted betweenness centrality:
        Krackhardt kite graph normalized
        """
    G = nx.krackhardt_kite_graph()
    b_answer = {0: 0.023, 1: 0.023, 2: 0.0, 3: 0.102, 4: 0.0, 5: 0.231, 6: 0.231, 7: 0.389, 8: 0.222, 9: 0.0}
    b = nx.betweenness_centrality(G, weight='weight', normalized=True)
    for n in sorted(G):
        assert b[n] == pytest.approx(b_answer[n], abs=0.001)