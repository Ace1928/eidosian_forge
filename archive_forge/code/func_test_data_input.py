import pytest
import networkx as nx
from networkx.utils import nodes_equal
from .test_graph import BaseAttrGraphTester, BaseGraphTester
from .test_graph import TestEdgeSubgraph as _TestGraphEdgeSubgraph
from .test_graph import TestGraph as _TestGraph
def test_data_input(self):
    G = self.Graph({1: [2], 2: [1]}, name='test')
    assert G.name == 'test'
    assert sorted(G.adj.items()) == [(1, {2: {}}), (2, {1: {}})]
    assert sorted(G.succ.items()) == [(1, {2: {}}), (2, {1: {}})]
    assert sorted(G.pred.items()) == [(1, {2: {}}), (2, {1: {}})]