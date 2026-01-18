import math
import pytest
import networkx as nx
from networkx.generators.interval_graph import interval_graph
from networkx.utils import edges_equal
def test_interval_graph_0(self):
    intervals = [(1, 2), (1, 3)]
    expected_graph = nx.Graph()
    expected_graph.add_edge(*intervals)
    actual_g = interval_graph(intervals)
    assert set(actual_g.nodes) == set(expected_graph.nodes)
    assert edges_equal(expected_graph, actual_g)