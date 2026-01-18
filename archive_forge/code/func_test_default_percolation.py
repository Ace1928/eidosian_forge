import pytest
import networkx as nx
def test_default_percolation():
    G = nx.erdos_renyi_graph(42, 0.42, seed=42)
    assert nx.percolation_centrality(G) == pytest.approx(nx.betweenness_centrality(G))