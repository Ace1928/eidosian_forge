import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.utils import edges_equal, nodes_equal
def test_nbunch_is_list(self):
    G = self.G()
    G.add_nodes_from(list('ABCD'))
    G.add_nodes_from(self.P3)
    assert sorted(G.nodes(), key=str) == [1, 2, 3, 'A', 'B', 'C', 'D']
    G.remove_nodes_from(self.P3)
    assert sorted(G.nodes(), key=str) == ['A', 'B', 'C', 'D']