import pytest
import networkx as nx
from networkx.utils import nodes_equal
from .test_graph import BaseAttrGraphTester, BaseGraphTester
from .test_graph import TestEdgeSubgraph as _TestGraphEdgeSubgraph
from .test_graph import TestGraph as _TestGraph
def test_has_successor(self):
    G = self.K3
    assert G.has_successor(0, 1)
    assert not G.has_successor(0, -1)