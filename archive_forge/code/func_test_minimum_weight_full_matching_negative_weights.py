import itertools
import pytest
import networkx as nx
from networkx.algorithms.bipartite.matching import (
def test_minimum_weight_full_matching_negative_weights(self):
    G = nx.complete_bipartite_graph(2, 2)
    G.add_edge(0, 2, weight=-2)
    G.add_edge(0, 3, weight=0.2)
    G.add_edge(1, 2, weight=-2)
    G.add_edge(1, 3, weight=0.3)
    matching = minimum_weight_full_matching(G)
    assert matching == {0: 3, 1: 2, 2: 1, 3: 0}