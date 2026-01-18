import pytest
import networkx as nx
def test_k5_load(self):
    G = self.K5
    c = nx.load_centrality(G)
    d = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
    for n in sorted(G):
        assert c[n] == pytest.approx(d[n], abs=0.001)