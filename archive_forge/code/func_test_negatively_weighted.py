import pytest
import networkx as nx
def test_negatively_weighted(self):
    with pytest.raises(nx.NetworkXError):
        G = nx.Graph()
        G.add_weighted_edges_from([(0, 1, -2), (1, 2, +1)])
        nx.local_reaching_centrality(G, 0, weight='weight')