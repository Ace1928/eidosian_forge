import pytest
import networkx as nx
from networkx.convert import (
from networkx.generators.classic import barbell_graph, cycle_graph
from networkx.utils import edges_equal, graphs_equal, nodes_equal
def test_custom_node_attr_dict_safekeeping(self):

    class custom_dict(dict):
        pass

    class Custom(nx.Graph):
        node_attr_dict_factory = custom_dict
    g = nx.Graph()
    g.add_node(1, weight=1)
    h = Custom(g)
    assert isinstance(g._node[1], dict)
    assert isinstance(h._node[1], custom_dict)