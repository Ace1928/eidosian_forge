import pytest
import networkx as nx
from networkx.algorithms.community import (
@pytest.mark.parametrize('func', (greedy_modularity_communities, naive_greedy_modularity_communities))
def test_modularity_communities_categorical_labels(func):
    G = nx.Graph([('a', 'b'), ('a', 'c'), ('b', 'c'), ('b', 'd'), ('d', 'e'), ('d', 'f'), ('d', 'g'), ('f', 'g'), ('d', 'e'), ('f', 'e')])
    expected = {frozenset({'f', 'g', 'e', 'd'}), frozenset({'a', 'b', 'c'})}
    assert set(func(G)) == expected