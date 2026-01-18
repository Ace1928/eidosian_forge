import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.utils import edges_equal, nodes_equal
def test_complete_subgraph(self):
    K1 = nx.complete_graph(1)
    K3 = nx.complete_graph(3)
    K5 = nx.complete_graph(5)
    H = K5.subgraph([1, 2, 3])
    assert nx.is_isomorphic(H, K3)