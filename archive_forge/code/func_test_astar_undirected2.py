import pytest
import networkx as nx
from networkx.utils import pairwise
def test_astar_undirected2(self):
    XG3 = nx.Graph()
    edges = [(0, 1, 2), (1, 2, 12), (2, 3, 1), (3, 4, 5), (4, 5, 1), (5, 0, 10)]
    XG3.add_weighted_edges_from(edges)
    assert nx.astar_path(XG3, 0, 3) == [0, 1, 2, 3]
    assert nx.astar_path_length(XG3, 0, 3) == 15