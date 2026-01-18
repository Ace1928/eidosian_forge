import pytest
import networkx as nx
def test_irreducible2(self):
    edges = [(1, 2), (2, 1), (2, 3), (3, 2), (4, 2), (4, 3), (5, 1), (6, 4), (6, 5)]
    G = nx.DiGraph(edges)
    assert nx.dominance_frontiers(G, 6) == {1: {2}, 2: {1, 3}, 3: {2}, 4: {2, 3}, 5: {1}, 6: set()}