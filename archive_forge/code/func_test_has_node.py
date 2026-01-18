import gc
import pickle
import platform
import weakref
import pytest
import networkx as nx
from networkx.utils import edges_equal, graphs_equal, nodes_equal
def test_has_node(self):
    G = self.K3
    assert G.has_node(1)
    assert not G.has_node(4)
    assert not G.has_node([])
    assert not G.has_node({1: 1})