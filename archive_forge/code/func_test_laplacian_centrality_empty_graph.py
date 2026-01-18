import pytest
import networkx as nx
def test_laplacian_centrality_empty_graph():
    G = nx.empty_graph(3)
    with pytest.raises(ZeroDivisionError):
        d = nx.laplacian_centrality(G, normalized=True)