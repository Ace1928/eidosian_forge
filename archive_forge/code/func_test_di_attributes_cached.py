import pytest
import networkx as nx
from networkx.utils import nodes_equal
from .test_graph import BaseAttrGraphTester, BaseGraphTester
from .test_graph import TestEdgeSubgraph as _TestGraphEdgeSubgraph
from .test_graph import TestGraph as _TestGraph
def test_di_attributes_cached(self):
    G = self.K3.copy()
    assert id(G.in_edges) == id(G.in_edges)
    assert id(G.out_edges) == id(G.out_edges)
    assert id(G.in_degree) == id(G.in_degree)
    assert id(G.out_degree) == id(G.out_degree)
    assert id(G.succ) == id(G.succ)
    assert id(G.pred) == id(G.pred)