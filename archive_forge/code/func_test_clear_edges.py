import pytest
import networkx as nx
from networkx.utils import nodes_equal
from .test_graph import BaseAttrGraphTester, BaseGraphTester
from .test_graph import TestEdgeSubgraph as _TestGraphEdgeSubgraph
from .test_graph import TestGraph as _TestGraph
def test_clear_edges(self):
    G = self.K3
    G.graph['name'] = 'K3'
    nodes = list(G.nodes)
    G.clear_edges()
    assert list(G.nodes) == nodes
    expected = {0: {}, 1: {}, 2: {}}
    assert G.succ == expected
    assert G.pred == expected
    assert list(G.edges) == []
    assert G.graph['name'] == 'K3'