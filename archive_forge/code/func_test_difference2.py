import os
import pytest
import networkx as nx
from networkx.classes.tests import dispatch_interface
from networkx.utils import edges_equal
def test_difference2():
    G = nx.Graph()
    H = nx.Graph()
    G.add_nodes_from([1, 2, 3, 4])
    H.add_nodes_from([1, 2, 3, 4])
    G.add_edge(1, 2)
    H.add_edge(1, 2)
    G.add_edge(2, 3)
    D = nx.difference(G, H)
    assert set(D.nodes()) == {1, 2, 3, 4}
    assert sorted(D.edges()) == [(2, 3)]
    D = nx.difference(H, G)
    assert set(D.nodes()) == {1, 2, 3, 4}
    assert sorted(D.edges()) == []
    H.add_edge(3, 4)
    D = nx.difference(H, G)
    assert set(D.nodes()) == {1, 2, 3, 4}
    assert sorted(D.edges()) == [(3, 4)]