import pytest
import networkx as nx
def test_k5_edge_load(self):
    G = self.K5
    c = nx.edge_load_centrality(G)
    d = {(0, 1): 5.0, (0, 2): 5.0, (0, 3): 5.0, (0, 4): 5.0, (1, 2): 5.0, (1, 3): 5.0, (1, 4): 5.0, (2, 3): 5.0, (2, 4): 5.0, (3, 4): 5.0}
    for n in G.edges():
        assert c[n] == pytest.approx(d[n], abs=0.001)