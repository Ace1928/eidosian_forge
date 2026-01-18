import os
import pytest
import networkx as nx
from networkx.classes.tests import dispatch_interface
from networkx.utils import edges_equal
def test_intersection():
    G = nx.Graph()
    H = nx.Graph()
    G.add_nodes_from([1, 2, 3, 4])
    G.add_edge(1, 2)
    G.add_edge(2, 3)
    H.add_nodes_from([1, 2, 3, 4])
    H.add_edge(2, 3)
    H.add_edge(3, 4)
    I = nx.intersection(G, H)
    assert set(I.nodes()) == {1, 2, 3, 4}
    assert sorted(I.edges()) == [(2, 3)]
    G2 = dispatch_interface.convert(G)
    H2 = dispatch_interface.convert(H)
    I2 = nx.intersection(G2, H2)
    assert set(I2.nodes()) == {1, 2, 3, 4}
    assert sorted(I2.edges()) == [(2, 3)]
    if not nx.utils.backends._dispatch._automatic_backends:
        with pytest.raises(TypeError):
            nx.intersection(G2, H)
        with pytest.raises(TypeError):
            nx.intersection(G, H2)