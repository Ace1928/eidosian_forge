import pytest
import networkx as nx
def test_indegree_centrality(self):
    d = nx.in_degree_centrality(self.G)
    exact = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.625, 6: 0.125, 7: 0.125, 8: 0.125}
    for n, dc in d.items():
        assert exact[n] == pytest.approx(dc, abs=1e-07)