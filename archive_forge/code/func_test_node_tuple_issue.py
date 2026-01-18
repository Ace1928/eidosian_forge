import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.utils import edges_equal, nodes_equal
def test_node_tuple_issue(self):
    H = self.G()
    pytest.raises(nx.NetworkXError, H.remove_node, (1, 2))
    H.remove_nodes_from([(1, 2)])
    pytest.raises(nx.NetworkXError, H.neighbors, (1, 2))