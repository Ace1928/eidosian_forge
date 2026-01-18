import pytest
import networkx as nx
from networkx.utils import pairwise
def test_cycle(self):
    C = nx.cycle_graph(7)
    assert nx.astar_path(C, 0, 3) == [0, 1, 2, 3]
    assert nx.dijkstra_path(C, 0, 4) == [0, 6, 5, 4]