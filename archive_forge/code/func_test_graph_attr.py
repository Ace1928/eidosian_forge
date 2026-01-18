import gc
import pickle
import platform
import weakref
import pytest
import networkx as nx
from networkx.utils import edges_equal, graphs_equal, nodes_equal
def test_graph_attr(self):
    G = self.K3.copy()
    G.graph['foo'] = 'bar'
    assert isinstance(G.graph, G.graph_attr_dict_factory)
    assert G.graph['foo'] == 'bar'
    del G.graph['foo']
    assert G.graph == {}
    H = self.Graph(foo='bar')
    assert H.graph['foo'] == 'bar'