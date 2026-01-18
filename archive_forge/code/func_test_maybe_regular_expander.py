import pytest
import networkx as nx
@pytest.mark.parametrize('d, n', [(2, 7), (4, 10), (4, 16)])
def test_maybe_regular_expander(d, n):
    pytest.importorskip('numpy')
    G = nx.maybe_regular_expander(n, d)
    assert len(G) == n, 'Should have n nodes'
    assert len(G.edges) == n * d / 2, 'Should have n*d/2 edges'
    assert nx.is_k_regular(G, d), 'Should be d-regular'