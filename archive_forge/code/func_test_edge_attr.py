from collections import UserDict
import pytest
import networkx as nx
from networkx.utils import edges_equal
from .test_graph import BaseAttrGraphTester
from .test_graph import TestGraph as _TestGraph
def test_edge_attr(self):
    G = self.Graph()
    G.add_edge(1, 2, key='k1', foo='bar')
    G.add_edge(1, 2, key='k2', foo='baz')
    assert isinstance(G.get_edge_data(1, 2), G.edge_key_dict_factory)
    assert all((isinstance(d, G.edge_attr_dict_factory) for u, v, d in G.edges(data=True)))
    assert edges_equal(G.edges(keys=True, data=True), [(1, 2, 'k1', {'foo': 'bar'}), (1, 2, 'k2', {'foo': 'baz'})])
    assert edges_equal(G.edges(keys=True, data='foo'), [(1, 2, 'k1', 'bar'), (1, 2, 'k2', 'baz')])