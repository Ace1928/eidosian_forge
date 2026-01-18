import pytest
import networkx as nx
from networkx.convert import (
from networkx.generators.classic import barbell_graph, cycle_graph
from networkx.utils import edges_equal, graphs_equal, nodes_equal
def test_attribute_dict_integrity(self):
    G = nx.Graph()
    G.add_nodes_from('abc')
    H = to_networkx_graph(G, create_using=nx.Graph)
    assert list(H.nodes) == list(G.nodes)
    H = nx.DiGraph(G)
    assert list(H.nodes) == list(G.nodes)