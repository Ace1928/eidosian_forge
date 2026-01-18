import pytest
import networkx as nx
def test_bridges_multiple_components(self):
    G = nx.Graph()
    nx.add_path(G, [0, 1, 2])
    nx.add_path(G, [4, 5, 6])
    assert list(nx.bridges(G, root=4)) == [(4, 5), (5, 6)]