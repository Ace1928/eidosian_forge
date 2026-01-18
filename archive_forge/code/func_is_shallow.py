from collections import UserDict
import pytest
import networkx as nx
from networkx.utils import edges_equal
from .test_multigraph import BaseMultiGraphTester
from .test_multigraph import TestEdgeSubgraph as _TestMultiGraphEdgeSubgraph
from .test_multigraph import TestMultiGraph as _TestMultiGraph
def is_shallow(self, H, G):
    assert G.graph['foo'] == H.graph['foo']
    G.graph['foo'].append(1)
    assert G.graph['foo'] == H.graph['foo']
    assert G.nodes[0]['foo'] == H.nodes[0]['foo']
    G.nodes[0]['foo'].append(1)
    assert G.nodes[0]['foo'] == H.nodes[0]['foo']
    assert G[1][2][0]['foo'] == H[1][2][0]['foo']
    G[1][2][0]['foo'].append(1)
    assert G[1][2][0]['foo'] == H[1][2][0]['foo']