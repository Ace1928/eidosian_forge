import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.utils import edges_equal, nodes_equal
def test_nbunch_dict(self):
    G = self.G()
    nbunch = set('ABCDEFGHIJKL')
    G.add_nodes_from(nbunch)
    nbunch = {'I': 'foo', 'J': 2, 'K': True, 'L': 'spam'}
    G.remove_nodes_from(nbunch)
    assert sorted(G.nodes(), key=str), ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']