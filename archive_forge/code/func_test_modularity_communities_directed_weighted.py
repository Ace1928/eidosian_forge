import pytest
import networkx as nx
from networkx.algorithms.community import (
def test_modularity_communities_directed_weighted():
    G = nx.DiGraph()
    G.add_weighted_edges_from([(1, 2, 5), (1, 3, 3), (2, 3, 6), (2, 6, 1), (1, 4, 1), (4, 5, 3), (4, 6, 7), (5, 6, 2), (5, 7, 5), (5, 8, 4), (6, 8, 3)])
    expected = [frozenset({4, 5, 6, 7, 8}), frozenset({1, 2, 3})]
    assert greedy_modularity_communities(G, weight='weight') == expected
    G[2][6]['weight'] = 20
    expected = [frozenset({1, 2, 3, 6}), frozenset({4, 5, 7, 8})]
    assert greedy_modularity_communities(G, weight='weight') == expected