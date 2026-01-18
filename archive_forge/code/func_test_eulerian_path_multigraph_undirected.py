import collections
import pytest
import networkx as nx
def test_eulerian_path_multigraph_undirected(self):
    G = nx.MultiGraph()
    result = [(2, 1), (1, 2), (2, 1), (1, 2), (2, 3), (3, 4)]
    G.add_edges_from(result)
    assert result == list(nx.eulerian_path(G))
    assert result == list(nx.eulerian_path(G, source=2))
    with pytest.raises(nx.NetworkXError):
        list(nx.eulerian_path(G, source=3))
    with pytest.raises(nx.NetworkXError):
        list(nx.eulerian_path(G, source=1))