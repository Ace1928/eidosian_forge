import itertools
import pytest
import networkx as nx
from networkx.algorithms.bipartite.matching import (
def test_minimum_weight_full_matching_different_weight_key(self):
    G = nx.complete_bipartite_graph(2, 2)
    G.add_edge(0, 2, mass=2)
    G.add_edge(0, 3, mass=0.2)
    G.add_edge(1, 2, mass=1)
    G.add_edge(1, 3, mass=2)
    matching = minimum_weight_full_matching(G, weight='mass')
    assert matching == {0: 3, 1: 2, 2: 1, 3: 0}