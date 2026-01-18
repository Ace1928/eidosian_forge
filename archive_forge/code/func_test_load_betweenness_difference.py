import pytest
import networkx as nx
def test_load_betweenness_difference(self):
    B = nx.Graph()
    B.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3), (2, 4), (4, 5), (3, 5)])
    c = nx.load_centrality(B, normalized=False)
    d = {0: 1.75, 1: 1.75, 2: 6.5, 3: 6.5, 4: 1.75, 5: 1.75}
    for n in sorted(B):
        assert c[n] == pytest.approx(d[n], abs=0.001)