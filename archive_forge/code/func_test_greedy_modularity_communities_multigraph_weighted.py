import pytest
import networkx as nx
from networkx.algorithms.community import (
def test_greedy_modularity_communities_multigraph_weighted():
    G = nx.MultiGraph()
    G.add_weighted_edges_from([(1, 2, 5), (1, 2, 3), (1, 3, 6), (1, 3, 6), (2, 3, 4), (1, 4, 1), (1, 4, 1), (2, 4, 3), (2, 4, 3), (4, 5, 1), (5, 6, 3), (5, 6, 7), (5, 6, 4), (5, 7, 9), (5, 7, 9), (6, 7, 8), (7, 8, 2), (7, 8, 2), (5, 8, 6), (5, 8, 6)])
    expected = [frozenset({1, 2, 3, 4}), frozenset({5, 6, 7, 8})]
    assert greedy_modularity_communities(G, weight='weight') == expected
    G.add_edge(4, 5, weight=16)
    expected = [frozenset({4, 5, 6, 7, 8}), frozenset({1, 2, 3})]
    assert greedy_modularity_communities(G, weight='weight') == expected
    G[1][4][1]['weight'] = 3
    expected = [frozenset({1, 2, 3, 4}), frozenset({5, 6, 7, 8})]
    assert greedy_modularity_communities(G, weight='weight') == expected