import math
import pytest
import networkx as nx
from networkx.generators.interval_graph import interval_graph
from networkx.utils import edges_equal
def test_interval_graph_4(self):
    """test all possible overlaps"""
    intervals = [(0, 2), (-2, -1), (-2, 0), (-2, 1), (-2, 2), (-2, 3), (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3), (3, 4)]
    expected_graph = nx.Graph()
    expected_graph.add_nodes_from(intervals)
    expected_nbrs = {(-2, 0), (-2, 1), (-2, 2), (-2, 3), (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)}
    actual_g = nx.interval_graph(intervals)
    actual_nbrs = nx.neighbors(actual_g, (0, 2))
    assert set(actual_nbrs) == expected_nbrs