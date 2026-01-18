import collections
import pytest
import networkx as nx
def test_eulerian_path_straight_link(self):
    G = nx.DiGraph()
    result = [(1, 2), (2, 3), (3, 4), (4, 5)]
    G.add_edges_from(result)
    assert result == list(nx.eulerian_path(G))
    assert result == list(nx.eulerian_path(G, source=1))
    with pytest.raises(nx.NetworkXError):
        list(nx.eulerian_path(G, source=3))
    with pytest.raises(nx.NetworkXError):
        list(nx.eulerian_path(G, source=4))
    with pytest.raises(nx.NetworkXError):
        list(nx.eulerian_path(G, source=5))