import pytest
import networkx as nx
def test_c4_edge_load(self):
    G = self.C4
    c = nx.edge_load_centrality(G)
    d = {(0, 1): 6.0, (0, 3): 6.0, (1, 2): 6.0, (2, 3): 6.0}
    for n in G.edges():
        assert c[n] == pytest.approx(d[n], abs=0.001)