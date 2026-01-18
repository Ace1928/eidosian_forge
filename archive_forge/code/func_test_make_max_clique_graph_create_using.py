import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
def test_make_max_clique_graph_create_using(self):
    G = nx.Graph([(1, 2), (3, 1), (4, 1), (5, 6)])
    E = nx.Graph([(0, 1), (0, 2), (1, 2)])
    E.add_node(3)
    assert nx.is_isomorphic(nx.make_max_clique_graph(G, create_using=nx.Graph), E)