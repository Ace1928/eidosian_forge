import pytest
import networkx as nx
def test_degree_sequence_tree():
    z = [1, 1, 1, 1, 1, 2, 2, 2, 3, 4]
    G = nx.degree_sequence_tree(z)
    assert len(G) == len(z)
    assert len(list(G.edges())) == sum(z) / 2
    pytest.raises(nx.NetworkXError, nx.degree_sequence_tree, z, create_using=nx.DiGraph())
    z = [1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 4]
    pytest.raises(nx.NetworkXError, nx.degree_sequence_tree, z)