import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.utils import edges_equal, nodes_equal
def test_add_edges_from3(self):
    G = self.G()
    G.add_edges_from(zip(list('ACD'), list('CDE')))
    assert G.has_edge('D', 'E')
    assert not G.has_edge('E', 'C')