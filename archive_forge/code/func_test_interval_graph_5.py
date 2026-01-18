import math
import pytest
import networkx as nx
from networkx.generators.interval_graph import interval_graph
from networkx.utils import edges_equal
def test_interval_graph_5(self):
    """this test is to see that an interval supports infinite number"""
    intervals = {(-math.inf, 0), (-1, -1), (0.5, 0.5), (1, 1), (1, math.inf)}
    expected_graph = nx.Graph()
    expected_graph.add_nodes_from(intervals)
    e1 = ((-math.inf, 0), (-1, -1))
    e2 = ((1, 1), (1, math.inf))
    expected_graph.add_edges_from([e1, e2])
    actual_g = interval_graph(intervals)
    assert set(actual_g.nodes) == set(expected_graph.nodes)
    assert edges_equal(expected_graph, actual_g)