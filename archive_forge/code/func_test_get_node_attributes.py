import random
import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
def test_get_node_attributes():
    graphs = [nx.Graph(), nx.DiGraph(), nx.MultiGraph(), nx.MultiDiGraph()]
    for G in graphs:
        G = nx.path_graph(3, create_using=G)
        attr = 'hello'
        vals = 100
        nx.set_node_attributes(G, vals, attr)
        attrs = nx.get_node_attributes(G, attr)
        assert attrs[0] == vals
        assert attrs[1] == vals
        assert attrs[2] == vals
        default_val = 1
        G.add_node(4)
        attrs = nx.get_node_attributes(G, attr, default=default_val)
        assert attrs[4] == default_val