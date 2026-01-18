import pytest
import networkx as nx
@pytest.mark.parametrize('generator', _gnp_generators)
@pytest.mark.parametrize('p', (0.2, 0.8))
@pytest.mark.parametrize('directed', (True, False))
def test_gnp_generators_edge_probability(generator, p, directed):
    """Test that gnp generators generate edges according to the their probability `p`."""
    runs = 5000
    n = 5
    edge_counts = [[0] * n for _ in range(n)]
    for i in range(runs):
        G = generator(n, p, directed=directed)
        for v, w in G.edges:
            edge_counts[v][w] += 1
            if not directed:
                edge_counts[w][v] += 1
    for v in range(n):
        for w in range(n):
            if v == w:
                assert edge_counts[v][w] == 0
            else:
                assert abs(edge_counts[v][w] / float(runs) - p) <= 0.03