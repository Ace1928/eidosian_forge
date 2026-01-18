import pytest
import networkx as nx
def test_no_perfect_matching_raises():
    G = nx.Graph([(0, 1), (0, 2)])
    with pytest.raises(nx.NetworkXError, match='.*not contain a perfect matching'):
        nx.bipartite.maximal_extendability(G)