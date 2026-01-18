import networkx as nx
from .base_test import BaseTestAttributeMixing, BaseTestDegreeMixing
def test_node_attribute_xy_directed(self):
    attrxy = sorted(nx.node_attribute_xy(self.D, 'fish'))
    attrxy_result = sorted([('one', 'one'), ('two', 'two'), ('one', 'red'), ('two', 'blue')])
    assert attrxy == attrxy_result