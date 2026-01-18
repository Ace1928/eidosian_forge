import collections
import pytest
import networkx as nx
@pytest.mark.parametrize('G', (nx.Graph(), nx.DiGraph()))
def test_has_eulerian_path_not_weakly_connected(self, G):
    G.add_edges_from([(0, 1), (2, 3), (3, 2)])
    assert not nx.has_eulerian_path(G)