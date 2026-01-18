import pytest
import networkx as nx
def test_non_positive_weights(self):
    with pytest.raises(nx.NetworkXError):
        G = nx.DiGraph()
        G.add_weighted_edges_from([(0, 1, 0)])
        nx.local_reaching_centrality(G, 0, weight='weight')