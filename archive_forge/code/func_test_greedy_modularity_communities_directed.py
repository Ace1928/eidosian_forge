import pytest
import networkx as nx
from networkx.algorithms.community import (
def test_greedy_modularity_communities_directed():
    G = nx.DiGraph([('a', 'b'), ('a', 'c'), ('b', 'c'), ('b', 'd'), ('d', 'e'), ('d', 'f'), ('d', 'g'), ('f', 'g'), ('d', 'e'), ('f', 'e')])
    expected = [frozenset({'f', 'g', 'e', 'd'}), frozenset({'a', 'b', 'c'})]
    assert greedy_modularity_communities(G) == expected
    G = nx.DiGraph()
    G.add_edges_from([(1, 1), (1, 2), (1, 3), (2, 3), (1, 4), (4, 4), (5, 5), (4, 5), (4, 6), (5, 6)])
    expected = [frozenset({1, 2, 3}), frozenset({4, 5, 6})]
    assert greedy_modularity_communities(G) == expected