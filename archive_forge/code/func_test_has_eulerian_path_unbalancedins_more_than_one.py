import collections
import pytest
import networkx as nx
@pytest.mark.parametrize('G', (nx.Graph(), nx.DiGraph()))
def test_has_eulerian_path_unbalancedins_more_than_one(self, G):
    G.add_edges_from([(0, 1), (2, 3)])
    assert not nx.has_eulerian_path(G)