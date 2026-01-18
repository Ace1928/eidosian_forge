from collections import UserDict
import pytest
import networkx as nx
from networkx.utils import edges_equal
from .test_graph import BaseAttrGraphTester
from .test_graph import TestGraph as _TestGraph
def test_get_edge_data(self):
    G = self.K3
    assert G.get_edge_data(0, 1) == {0: {}}
    assert G[0][1] == {0: {}}
    assert G[0][1][0] == {}
    assert G.get_edge_data(10, 20) is None
    assert G.get_edge_data(0, 1, 0) == {}