import pytest
import networkx as nx
from networkx.utils import nodes_equal
from .test_graph import BaseAttrGraphTester, BaseGraphTester
from .test_graph import TestEdgeSubgraph as _TestGraphEdgeSubgraph
from .test_graph import TestGraph as _TestGraph
def test_di_cache_reset(self):
    G = self.K3.copy()
    old_succ = G.succ
    assert id(G.succ) == id(old_succ)
    old_adj = G.adj
    assert id(G.adj) == id(old_adj)
    G._succ = {}
    assert id(G.succ) != id(old_succ)
    assert id(G.adj) != id(old_adj)
    old_pred = G.pred
    assert id(G.pred) == id(old_pred)
    G._pred = {}
    assert id(G.pred) != id(old_pred)