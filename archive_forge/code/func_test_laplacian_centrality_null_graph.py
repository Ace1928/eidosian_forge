import pytest
import networkx as nx
def test_laplacian_centrality_null_graph():
    G = nx.Graph()
    with pytest.raises(nx.NetworkXPointlessConcept):
        d = nx.laplacian_centrality(G, normalized=False)