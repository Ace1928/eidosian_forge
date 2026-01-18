import networkx as nx
from .base_test import BaseTestAttributeMixing, BaseTestDegreeMixing
def test_node_degree_xy_undirected_nodes(self):
    xy = sorted(nx.node_degree_xy(self.P4, nodes=[0, 1, -1]))
    xy_result = sorted([(1, 2), (2, 1)])
    assert xy == xy_result