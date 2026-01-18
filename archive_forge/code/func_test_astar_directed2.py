import pytest
import networkx as nx
from networkx.utils import pairwise
def test_astar_directed2(self):
    XG2 = nx.DiGraph()
    edges = [(1, 4, 1), (4, 5, 1), (5, 6, 1), (6, 3, 1), (1, 3, 50), (1, 2, 100), (2, 3, 100)]
    XG2.add_weighted_edges_from(edges)
    assert nx.astar_path(XG2, 1, 3) == [1, 4, 5, 6, 3]