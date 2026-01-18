import pytest
import networkx as nx
from networkx.utils import pairwise
def test_all_pairs_dijkstra_path(self):
    cycle = nx.cycle_graph(7)
    p = dict(nx.all_pairs_dijkstra_path(cycle))
    assert p[0][3] == [0, 1, 2, 3]
    cycle[1][2]['weight'] = 10
    p = dict(nx.all_pairs_dijkstra_path(cycle))
    assert p[0][3] == [0, 6, 5, 4, 3]