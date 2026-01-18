from collections import UserDict
import pytest
import networkx as nx
from networkx.utils import edges_equal
from .test_graph import BaseAttrGraphTester
from .test_graph import TestGraph as _TestGraph
def test_edge_lookup(self):
    G = self.Graph()
    G.add_edge(1, 2, foo='bar')
    G.add_edge(1, 2, 'key', foo='biz')
    assert edges_equal(G.edges[1, 2, 0], {'foo': 'bar'})
    assert edges_equal(G.edges[1, 2, 'key'], {'foo': 'biz'})