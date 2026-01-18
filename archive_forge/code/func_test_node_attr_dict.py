import pytest
import networkx as nx
from networkx.utils import edges_equal
def test_node_attr_dict(self):
    """Tests that the node attribute dictionary of the two graphs is
        the same object.

        """
    for v in self.H:
        assert self.G.nodes[v] == self.H.nodes[v]
    self.G.nodes[0]['name'] = 'foo'
    assert self.G.nodes[0] == self.H.nodes[0]
    self.H.nodes[1]['name'] = 'bar'
    assert self.G.nodes[1] == self.H.nodes[1]
    self.G.nodes[0]['name'] = 'node0'
    self.H.nodes[1]['name'] = 'node1'