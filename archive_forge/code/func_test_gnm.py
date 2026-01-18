import pytest
import networkx as nx
def test_gnm(self):
    G = nx.gnm_random_graph(10, 3)
    assert len(G) == 10
    assert G.number_of_edges() == 3
    G = nx.gnm_random_graph(10, 3, seed=42)
    assert len(G) == 10
    assert G.number_of_edges() == 3
    G = nx.gnm_random_graph(10, 100)
    assert len(G) == 10
    assert G.number_of_edges() == 45
    G = nx.gnm_random_graph(10, 100, directed=True)
    assert len(G) == 10
    assert G.number_of_edges() == 90
    G = nx.gnm_random_graph(10, -1.1)
    assert len(G) == 10
    assert G.number_of_edges() == 0