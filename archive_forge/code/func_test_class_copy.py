import gc
import pickle
import platform
import weakref
import pytest
import networkx as nx
from networkx.utils import edges_equal, graphs_equal, nodes_equal
def test_class_copy(self):
    G = self.Graph()
    G.add_node(0)
    G.add_edge(1, 2)
    self.add_attributes(G)
    H = G.__class__(G)
    self.graphs_equal(H, G)
    self.different_attrdict(H, G)
    self.shallow_copy_attrdict(H, G)