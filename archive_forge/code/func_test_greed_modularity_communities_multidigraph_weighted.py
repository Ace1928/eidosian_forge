import pytest
import networkx as nx
from networkx.algorithms.community import (
def test_greed_modularity_communities_multidigraph_weighted():
    G = nx.MultiDiGraph()
    G.add_weighted_edges_from([(1, 2, 5), (1, 2, 3), (3, 1, 6), (1, 3, 6), (3, 2, 4), (1, 4, 2), (1, 4, 5), (2, 4, 3), (3, 2, 8), (4, 2, 3), (4, 3, 5), (4, 5, 2), (5, 6, 3), (5, 6, 7), (6, 5, 4), (5, 7, 9), (5, 7, 9), (7, 6, 8), (7, 8, 2), (8, 7, 2), (5, 8, 6), (5, 8, 6)])
    expected = [frozenset({1, 2, 3, 4}), frozenset({5, 6, 7, 8})]
    assert greedy_modularity_communities(G, weight='weight') == expected