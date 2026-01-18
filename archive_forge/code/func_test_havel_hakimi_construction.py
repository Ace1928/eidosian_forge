import pytest
import networkx as nx
def test_havel_hakimi_construction():
    G = nx.havel_hakimi_graph([])
    assert len(G) == 0
    z = [1000, 3, 3, 3, 3, 2, 2, 2, 1, 1, 1]
    pytest.raises(nx.NetworkXError, nx.havel_hakimi_graph, z)
    z = ['A', 3, 3, 3, 3, 2, 2, 2, 1, 1, 1]
    pytest.raises(nx.NetworkXError, nx.havel_hakimi_graph, z)
    z = [5, 4, 3, 3, 3, 2, 2, 2]
    G = nx.havel_hakimi_graph(z)
    G = nx.configuration_model(z)
    z = [6, 5, 4, 4, 2, 1, 1, 1]
    pytest.raises(nx.NetworkXError, nx.havel_hakimi_graph, z)
    z = [10, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2]
    G = nx.havel_hakimi_graph(z)
    pytest.raises(nx.NetworkXError, nx.havel_hakimi_graph, z, create_using=nx.DiGraph())