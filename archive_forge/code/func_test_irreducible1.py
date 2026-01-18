import pytest
import networkx as nx
def test_irreducible1(self):
    edges = [(1, 2), (2, 1), (3, 2), (4, 1), (5, 3), (5, 4)]
    G = nx.DiGraph(edges)
    assert dict(nx.dominance_frontiers(G, 5).items()) == {1: {2}, 2: {1}, 3: {2}, 4: {1}, 5: set()}