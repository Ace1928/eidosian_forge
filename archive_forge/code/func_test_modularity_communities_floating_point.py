import pytest
import networkx as nx
from networkx.algorithms.community import (
def test_modularity_communities_floating_point():
    G = nx.Graph()
    G.add_weighted_edges_from([(0, 1, 12), (1, 4, 71), (2, 3, 15), (2, 4, 10), (3, 6, 13)])
    expected = [{0, 1, 4}, {2, 3, 6}]
    assert greedy_modularity_communities(G, weight='weight') == expected
    assert greedy_modularity_communities(G, weight='weight', resolution=0.99) == expected