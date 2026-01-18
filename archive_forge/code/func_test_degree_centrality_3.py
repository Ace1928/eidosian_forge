import pytest
import networkx as nx
def test_degree_centrality_3(self):
    d = nx.degree_centrality(self.K)
    exact = {0: 0.444, 1: 0.444, 2: 0.333, 3: 0.667, 4: 0.333, 5: 0.556, 6: 0.556, 7: 0.333, 8: 0.222, 9: 0.111}
    for n, dc in d.items():
        assert exact[n] == pytest.approx(float(f'{dc:.3f}'), abs=1e-07)