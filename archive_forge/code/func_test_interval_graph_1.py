import math
import pytest
import networkx as nx
from networkx.generators.interval_graph import interval_graph
from networkx.utils import edges_equal
def test_interval_graph_1(self):
    intervals = [(1, 2), (2, 3), (3, 4), (1, 4)]
    expected_graph = nx.Graph()
    expected_graph.add_nodes_from(intervals)
    e1 = ((1, 4), (1, 2))
    e2 = ((1, 4), (2, 3))
    e3 = ((1, 4), (3, 4))
    e4 = ((3, 4), (2, 3))
    e5 = ((1, 2), (2, 3))
    expected_graph.add_edges_from([e1, e2, e3, e4, e5])
    actual_g = interval_graph(intervals)
    assert set(actual_g.nodes) == set(expected_graph.nodes)
    assert edges_equal(expected_graph, actual_g)