import gc
import pickle
import platform
import weakref
import pytest
import networkx as nx
from networkx.utils import edges_equal, graphs_equal, nodes_equal
def test_node_attr(self):
    G = self.K3.copy()
    G.add_node(1, foo='bar')
    assert all((isinstance(d, G.node_attr_dict_factory) for u, d in G.nodes(data=True)))
    assert nodes_equal(G.nodes(), [0, 1, 2])
    assert nodes_equal(G.nodes(data=True), [(0, {}), (1, {'foo': 'bar'}), (2, {})])
    G.nodes[1]['foo'] = 'baz'
    assert nodes_equal(G.nodes(data=True), [(0, {}), (1, {'foo': 'baz'}), (2, {})])
    assert nodes_equal(G.nodes(data='foo'), [(0, None), (1, 'baz'), (2, None)])
    assert nodes_equal(G.nodes(data='foo', default='bar'), [(0, 'bar'), (1, 'baz'), (2, 'bar')])