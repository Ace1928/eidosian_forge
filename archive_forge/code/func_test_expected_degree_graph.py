import pytest
import networkx as nx
def test_expected_degree_graph():
    deg_seq = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
    G1 = nx.expected_degree_graph(deg_seq, seed=1000)
    assert len(G1) == 12
    G2 = nx.expected_degree_graph(deg_seq, seed=1000)
    assert nx.is_isomorphic(G1, G2)
    G1 = nx.expected_degree_graph(deg_seq, seed=10)
    G2 = nx.expected_degree_graph(deg_seq, seed=10)
    assert nx.is_isomorphic(G1, G2)