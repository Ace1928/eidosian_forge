import pytest
import networkx as nx
from networkx import NetworkXNotImplemented
from networkx import convert_node_labels_to_integers as cnlti
from networkx.classes.tests import dispatch_interface
def test_is_connected(self):
    assert nx.is_connected(self.grid)
    G = nx.Graph()
    G.add_nodes_from([1, 2])
    assert not nx.is_connected(G)