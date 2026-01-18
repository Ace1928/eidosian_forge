import pytest
import networkx as nx
from networkx.utils import pairwise
def test_find_negative_cycle_single_edge(self):
    G = nx.Graph()
    G.add_edge(0, 1, weight=-1)
    assert nx.find_negative_cycle(G, 1) == [1, 0, 1]