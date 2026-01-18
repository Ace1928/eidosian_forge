import pytest
import networkx as nx
def test_laplacian_centrality_P3():
    P3 = nx.path_graph(3)
    d = nx.laplacian_centrality(P3)
    exact = {0: 0.6, 1: 1.0, 2: 0.6}
    for n, dc in d.items():
        assert exact[n] == pytest.approx(dc, abs=1e-07)