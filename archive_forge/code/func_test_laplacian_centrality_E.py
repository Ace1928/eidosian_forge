import pytest
import networkx as nx
def test_laplacian_centrality_E():
    E = nx.Graph()
    E.add_weighted_edges_from([(0, 1, 4), (4, 5, 1), (0, 2, 2), (2, 1, 1), (1, 3, 2), (1, 4, 2)])
    d = nx.laplacian_centrality(E)
    exact = {0: 0.7, 1: 0.9, 2: 0.28, 3: 0.22, 4: 0.26, 5: 0.04}
    for n, dc in d.items():
        assert exact[n] == pytest.approx(dc, abs=1e-07)
    full_energy = 200
    dnn = nx.laplacian_centrality(E, normalized=False)
    for n, dc in dnn.items():
        assert exact[n] * full_energy == pytest.approx(dc, abs=1e-07)
    duw_nn = nx.laplacian_centrality(E, normalized=False, weight=None)
    print(duw_nn)
    exact_uw_nn = {0: 18, 1: 34, 2: 18, 3: 10, 4: 16, 5: 6}
    for n, dc in duw_nn.items():
        assert exact_uw_nn[n] == pytest.approx(dc, abs=1e-07)
    duw = nx.laplacian_centrality(E, weight=None)
    full_energy = 42
    for n, dc in duw.items():
        assert exact_uw_nn[n] / full_energy == pytest.approx(dc, abs=1e-07)