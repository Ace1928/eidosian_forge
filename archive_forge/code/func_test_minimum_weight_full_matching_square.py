import itertools
import pytest
import networkx as nx
from networkx.algorithms.bipartite.matching import (
def test_minimum_weight_full_matching_square(self):
    G = nx.complete_bipartite_graph(3, 3)
    G.add_edge(0, 3, weight=400)
    G.add_edge(0, 4, weight=150)
    G.add_edge(0, 5, weight=400)
    G.add_edge(1, 3, weight=400)
    G.add_edge(1, 4, weight=450)
    G.add_edge(1, 5, weight=600)
    G.add_edge(2, 3, weight=300)
    G.add_edge(2, 4, weight=225)
    G.add_edge(2, 5, weight=300)
    matching = minimum_weight_full_matching(G)
    assert matching == {0: 4, 1: 3, 2: 5, 4: 0, 3: 1, 5: 2}