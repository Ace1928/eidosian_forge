import collections
import pytest
import networkx as nx
def test_eulerian_path_undirected(self):
    G = nx.Graph()
    result = [(1, 2), (2, 3), (3, 4), (4, 5)]
    result2 = [(5, 4), (4, 3), (3, 2), (2, 1)]
    G.add_edges_from(result)
    assert list(nx.eulerian_path(G)) in (result, result2)
    assert result == list(nx.eulerian_path(G, source=1))
    assert result2 == list(nx.eulerian_path(G, source=5))
    with pytest.raises(nx.NetworkXError):
        list(nx.eulerian_path(G, source=3))
    with pytest.raises(nx.NetworkXError):
        list(nx.eulerian_path(G, source=2))