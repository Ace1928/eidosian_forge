import pytest
import networkx as nx
from networkx import NetworkXNotImplemented
from networkx import convert_node_labels_to_integers as cnlti
from networkx.classes.tests import dispatch_interface
def test_connected_mutability(self):
    G = self.grid
    seen = set()
    for component in nx.connected_components(G):
        assert len(seen & component) == 0
        seen.update(component)
        component.clear()