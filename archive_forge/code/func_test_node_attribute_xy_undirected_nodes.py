import networkx as nx
from .base_test import BaseTestAttributeMixing, BaseTestDegreeMixing
def test_node_attribute_xy_undirected_nodes(self):
    attrxy = sorted(nx.node_attribute_xy(self.G, 'fish', nodes=['one', 'yellow']))
    attrxy_result = sorted([])
    assert attrxy == attrxy_result