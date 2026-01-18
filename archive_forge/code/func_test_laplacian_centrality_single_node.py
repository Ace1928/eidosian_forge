import pytest
import networkx as nx
def test_laplacian_centrality_single_node():
    """See gh-6571"""
    G = nx.empty_graph(1)
    assert nx.laplacian_centrality(G, normalized=False) == {0: 0}
    with pytest.raises(ZeroDivisionError):
        nx.laplacian_centrality(G, normalized=True)