import pytest
import networkx as nx
@pytest.mark.parametrize('generator', [nx.gnp_random_graph, nx.binomial_graph, nx.erdos_renyi_graph])
@pytest.mark.parametrize(('seed', 'directed', 'expected_num_edges'), [(42, False, 1219), (42, True, 2454), (314, False, 1247), (314, True, 2476)])
def test_gnp_random_graph_aliases(generator, seed, directed, expected_num_edges):
    """Test that aliases give the same result with the same seed."""
    G = generator(100, 0.25, seed=seed, directed=directed)
    assert len(G) == 100
    assert G.number_of_edges() == expected_num_edges
    assert G.is_directed() == directed