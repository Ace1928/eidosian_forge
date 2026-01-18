import gc
import pickle
import platform
import weakref
import pytest
import networkx as nx
from networkx.utils import edges_equal, graphs_equal, nodes_equal
def test_edge_attr2(self):
    G = self.Graph()
    G.add_edges_from([(1, 2), (3, 4)], foo='foo')
    assert edges_equal(G.edges(data=True), [(1, 2, {'foo': 'foo'}), (3, 4, {'foo': 'foo'})])
    assert edges_equal(G.edges(data='foo'), [(1, 2, 'foo'), (3, 4, 'foo')])