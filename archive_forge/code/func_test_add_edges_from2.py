import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.utils import edges_equal, nodes_equal
def test_add_edges_from2(self):
    G = self.G()
    G.add_edges_from([tuple('IJ'), list('KK'), tuple('JK')])
    assert G.has_edge(*('I', 'J'))
    assert G.has_edge(*('K', 'K'))
    assert G.has_edge(*('J', 'K'))
    if G.is_directed():
        assert not G.has_edge(*('K', 'J'))
    else:
        assert G.has_edge(*('K', 'J'))