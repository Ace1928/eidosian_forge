import pytest
import networkx as nx
def test_P2_normalized_load(self):
    G = self.P2
    c = nx.load_centrality(G, normalized=True)
    d = {0: 0.0, 1: 0.0}
    for n in sorted(G):
        assert c[n] == pytest.approx(d[n], abs=0.001)