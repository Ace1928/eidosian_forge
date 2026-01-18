from collections import UserDict
import pytest
import networkx as nx
from networkx.utils import edges_equal
from .test_multigraph import BaseMultiGraphTester
from .test_multigraph import TestEdgeSubgraph as _TestMultiGraphEdgeSubgraph
from .test_multigraph import TestMultiGraph as _TestMultiGraph
def test_in_edges(self):
    G = self.K3
    assert sorted(G.in_edges()) == [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
    assert sorted(G.in_edges(0)) == [(1, 0), (2, 0)]
    pytest.raises((KeyError, nx.NetworkXError), G.in_edges, -1)
    G.add_edge(0, 1, 2)
    assert sorted(G.in_edges()) == [(0, 1), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
    assert sorted(G.in_edges(0, keys=True)) == [(1, 0, 0), (2, 0, 0)]