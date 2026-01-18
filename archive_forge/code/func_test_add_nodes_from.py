import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.utils import edges_equal, nodes_equal
def test_add_nodes_from(self):
    G = self.G()
    G.add_nodes_from(list('ABCDEFGHIJKL'))
    assert G.has_node('L')
    G.remove_nodes_from(['H', 'I', 'J', 'K', 'L'])
    G.add_nodes_from([1, 2, 3, 4])
    assert sorted(G.nodes(), key=str) == [1, 2, 3, 4, 'A', 'B', 'C', 'D', 'E', 'F', 'G']
    assert sorted(G, key=str) == [1, 2, 3, 4, 'A', 'B', 'C', 'D', 'E', 'F', 'G']