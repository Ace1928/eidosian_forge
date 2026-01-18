from collections import UserDict
import pytest
import networkx as nx
from networkx.utils import edges_equal
from .test_graph import BaseAttrGraphTester
from .test_graph import TestGraph as _TestGraph
def test_edge_attr4(self):
    G = self.Graph()
    G.add_edge(1, 2, key=0, data=7, spam='bar', bar='foo')
    assert edges_equal(G.edges(data=True), [(1, 2, {'data': 7, 'spam': 'bar', 'bar': 'foo'})])
    G[1][2][0]['data'] = 10
    assert edges_equal(G.edges(data=True), [(1, 2, {'data': 10, 'spam': 'bar', 'bar': 'foo'})])
    G.adj[1][2][0]['data'] = 20
    assert edges_equal(G.edges(data=True), [(1, 2, {'data': 20, 'spam': 'bar', 'bar': 'foo'})])
    G.edges[1, 2, 0]['data'] = 21
    assert edges_equal(G.edges(data=True), [(1, 2, {'data': 21, 'spam': 'bar', 'bar': 'foo'})])
    G.adj[1][2][0]['listdata'] = [20, 200]
    G.adj[1][2][0]['weight'] = 20
    assert edges_equal(G.edges(data=True), [(1, 2, {'data': 21, 'spam': 'bar', 'bar': 'foo', 'listdata': [20, 200], 'weight': 20})])