import networkx as nx
from .base_test import BaseTestAttributeMixing, BaseTestDegreeMixing
def test_node_attribute_xy_selfloop(self):
    attrxy = sorted(nx.node_attribute_xy(self.S, 'fish'))
    attrxy_result = [('one', 'one'), ('two', 'two')]
    assert attrxy == attrxy_result