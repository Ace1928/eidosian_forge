import pytest
import networkx as nx
def test_expected_degree_graph_skew():
    deg_seq = [10, 2, 2, 2, 2]
    G1 = nx.expected_degree_graph(deg_seq, seed=1000)
    G2 = nx.expected_degree_graph(deg_seq, seed=1000)
    assert nx.is_isomorphic(G1, G2)
    assert len(G1) == 5