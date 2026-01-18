import random
import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
def test_edge_attributes_are_still_mutable_on_frozen_graph(self):
    G = nx.freeze(nx.path_graph(3))
    edge = G.edges[0, 1]
    edge['edge_attribute'] = True
    assert edge['edge_attribute'] == True