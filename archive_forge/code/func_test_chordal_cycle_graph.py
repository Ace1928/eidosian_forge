import pytest
import networkx as nx
@pytest.mark.parametrize('p', (3, 5, 7, 11))
def test_chordal_cycle_graph(p):
    """Test for the :func:`networkx.chordal_cycle_graph` function."""
    G = nx.chordal_cycle_graph(p)
    assert len(G) == p