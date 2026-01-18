import pytest
import networkx as nx
def test_attr_matrix_directed():
    G = nx.DiGraph()
    G.add_edge(0, 1, thickness=1, weight=3)
    G.add_edge(0, 1, thickness=1, weight=3)
    G.add_edge(0, 2, thickness=2)
    G.add_edge(1, 2, thickness=3)
    M = nx.attr_matrix(G, rc_order=[0, 1, 2])
    data = np.array([[0.0, 1.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
    np.testing.assert_equal(M, np.array(data))