from collections import UserDict
import pytest
import networkx as nx
from networkx.utils import edges_equal
from .test_graph import BaseAttrGraphTester
from .test_graph import TestGraph as _TestGraph
def test_add_edge_conflicting_key(self):
    G = self.Graph()
    G.add_edge(0, 1, key=1)
    G.add_edge(0, 1)
    assert G.number_of_edges() == 2
    G = self.Graph()
    G.add_edges_from([(0, 1, 1, {})])
    G.add_edges_from([(0, 1)])
    assert G.number_of_edges() == 2