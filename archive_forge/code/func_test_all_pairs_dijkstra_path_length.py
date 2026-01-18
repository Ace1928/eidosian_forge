import pytest
import networkx as nx
from networkx.utils import pairwise
def test_all_pairs_dijkstra_path_length(self):
    cycle = nx.cycle_graph(7)
    pl = dict(nx.all_pairs_dijkstra_path_length(cycle))
    assert pl[0] == {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 2, 6: 1}
    cycle[1][2]['weight'] = 10
    pl = dict(nx.all_pairs_dijkstra_path_length(cycle))
    assert pl[0] == {0: 0, 1: 1, 2: 5, 3: 4, 4: 3, 5: 2, 6: 1}