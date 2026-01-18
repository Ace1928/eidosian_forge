import networkx as nx
from .base_test import BaseTestAttributeMixing, BaseTestDegreeMixing
def test_node_degree_xy_weighted(self):
    G = nx.Graph()
    G.add_edge(1, 2, weight=7)
    G.add_edge(2, 3, weight=10)
    xy = sorted(nx.node_degree_xy(G, weight='weight'))
    xy_result = sorted([(7, 17), (17, 10), (17, 7), (10, 17)])
    assert xy == xy_result