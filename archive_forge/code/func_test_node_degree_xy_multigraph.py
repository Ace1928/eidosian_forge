import networkx as nx
from .base_test import BaseTestAttributeMixing, BaseTestDegreeMixing
def test_node_degree_xy_multigraph(self):
    xy = sorted(nx.node_degree_xy(self.M))
    xy_result = sorted([(2, 3), (2, 3), (3, 2), (3, 2), (2, 3), (3, 2), (1, 2), (2, 1)])
    assert xy == xy_result