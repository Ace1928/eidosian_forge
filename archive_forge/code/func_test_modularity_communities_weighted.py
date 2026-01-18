import pytest
import networkx as nx
from networkx.algorithms.community import (
@pytest.mark.parametrize('func', (greedy_modularity_communities, naive_greedy_modularity_communities))
def test_modularity_communities_weighted(func):
    G = nx.balanced_tree(2, 3)
    for a, b in G.edges:
        if (a == 1 or a == 2) and b != 0:
            G[a][b]['weight'] = 10.0
        else:
            G[a][b]['weight'] = 1.0
    expected = [{0, 1, 3, 4, 7, 8, 9, 10}, {2, 5, 6, 11, 12, 13, 14}]
    assert func(G, weight='weight') == expected
    assert func(G, weight='weight', resolution=0.9) == expected
    assert func(G, weight='weight', resolution=0.3) == expected
    assert func(G, weight='weight', resolution=1.1) != expected