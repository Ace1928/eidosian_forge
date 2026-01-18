import pytest
import networkx as nx
def test_boost_example(self):
    edges = [(0, 1), (1, 2), (1, 3), (2, 7), (3, 4), (4, 5), (4, 6), (5, 7), (6, 4)]
    G = nx.DiGraph(edges)
    assert nx.dominance_frontiers(G, 0) == {0: set(), 1: set(), 2: {7}, 3: {7}, 4: {4, 7}, 5: {7}, 6: {4}, 7: set()}
    result = nx.dominance_frontiers(G.reverse(copy=False), 7)
    expected = {0: set(), 1: set(), 2: {1}, 3: {1}, 4: {1, 4}, 5: {1}, 6: {4}, 7: set()}
    assert result == expected