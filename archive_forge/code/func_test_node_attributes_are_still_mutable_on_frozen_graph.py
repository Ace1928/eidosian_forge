import random
import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
def test_node_attributes_are_still_mutable_on_frozen_graph(self):
    G = nx.freeze(nx.path_graph(3))
    node = G.nodes[0]
    node['node_attribute'] = True
    assert node['node_attribute'] == True