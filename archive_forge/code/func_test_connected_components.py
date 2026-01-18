import pytest
import networkx as nx
from networkx import NetworkXNotImplemented
from networkx import convert_node_labels_to_integers as cnlti
from networkx.classes.tests import dispatch_interface
@pytest.mark.parametrize('wrapper', [lambda x: x, dispatch_interface.convert])
def test_connected_components(self, wrapper):
    cc = nx.connected_components
    G = wrapper(self.G)
    C = {frozenset([0, 1, 2, 3]), frozenset([4, 5, 6, 7, 8, 9]), frozenset([10, 11, 12, 13, 14])}
    assert {frozenset(g) for g in cc(G)} == C