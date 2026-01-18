import pytest
import networkx as nx
def test_caveman_graph():
    G = nx.caveman_graph(4, 3)
    assert len(G) == 12
    G = nx.caveman_graph(5, 1)
    E5 = nx.empty_graph(5)
    assert nx.is_isomorphic(G, E5)
    G = nx.caveman_graph(1, 5)
    K5 = nx.complete_graph(5)
    assert nx.is_isomorphic(G, K5)