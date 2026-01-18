import gc
import pickle
import platform
import weakref
import pytest
import networkx as nx
from networkx.utils import edges_equal, graphs_equal, nodes_equal
def test_attributes_cached(self):
    G = self.K3.copy()
    assert id(G.nodes) == id(G.nodes)
    assert id(G.edges) == id(G.edges)
    assert id(G.degree) == id(G.degree)
    assert id(G.adj) == id(G.adj)