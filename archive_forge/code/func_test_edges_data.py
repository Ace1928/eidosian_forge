import pytest
import networkx as nx
from networkx.utils import nodes_equal
from .test_graph import BaseAttrGraphTester, BaseGraphTester
from .test_graph import TestEdgeSubgraph as _TestGraphEdgeSubgraph
from .test_graph import TestGraph as _TestGraph
def test_edges_data(self):
    G = self.K3
    all_edges = [(0, 1, {}), (0, 2, {}), (1, 0, {}), (1, 2, {}), (2, 0, {}), (2, 1, {})]
    assert sorted(G.edges(data=True)) == all_edges
    assert sorted(G.edges(0, data=True)) == all_edges[:2]
    assert sorted(G.edges([0, 1], data=True)) == all_edges[:4]
    with pytest.raises(nx.NetworkXError):
        G.edges(-1, True)