import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.utils import edges_equal, nodes_equal
def test_nbunch_iterator(self):
    G = self.G()
    G.add_nodes_from(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'])
    n_iter = self.P3.nodes()
    G.add_nodes_from(n_iter)
    assert sorted(G.nodes(), key=str) == [1, 2, 3, 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    n_iter = self.P3.nodes()
    G.remove_nodes_from(n_iter)
    assert sorted(G.nodes(), key=str) == ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']