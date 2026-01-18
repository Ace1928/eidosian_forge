from collections import UserDict
import pytest
import networkx as nx
from networkx.utils import edges_equal
from .test_graph import BaseAttrGraphTester
from .test_graph import TestGraph as _TestGraph
def test_has_edge(self):
    G = self.K3
    assert G.has_edge(0, 1)
    assert not G.has_edge(0, -1)
    assert G.has_edge(0, 1, 0)
    assert not G.has_edge(0, 1, 1)