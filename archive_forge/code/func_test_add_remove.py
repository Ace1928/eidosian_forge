import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.utils import edges_equal, nodes_equal
def test_add_remove(self):
    G = self.G()
    G.add_node('m')
    assert G.has_node('m')
    G.add_node('m')
    pytest.raises(nx.NetworkXError, G.remove_node, 'j')
    G.remove_node('m')
    assert list(G) == []