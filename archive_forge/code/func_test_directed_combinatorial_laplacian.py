import pytest
import networkx as nx
from networkx.generators.degree_seq import havel_hakimi_graph
from networkx.generators.expanders import margulis_gabber_galil_graph
def test_directed_combinatorial_laplacian():
    """Directed combinatorial Laplacian"""
    G = nx.DiGraph()
    G.add_edges_from(((1, 2), (1, 3), (3, 1), (3, 2), (3, 5), (4, 5), (4, 6), (5, 4), (5, 6), (6, 4)))
    GL = np.array([[0.0366, -0.0132, -0.0153, -0.0034, -0.002, -0.0027], [-0.0132, 0.045, -0.0111, -0.0076, -0.0062, -0.0069], [-0.0153, -0.0111, 0.0408, -0.0035, -0.0083, -0.0027], [-0.0034, -0.0076, -0.0035, 0.3688, -0.1356, -0.2187], [-0.002, -0.0062, -0.0083, -0.1356, 0.2026, -0.0505], [-0.0027, -0.0069, -0.0027, -0.2187, -0.0505, 0.2815]])
    L = nx.directed_combinatorial_laplacian_matrix(G, alpha=0.9, nodelist=sorted(G))
    np.testing.assert_almost_equal(L, GL, decimal=3)
    G.add_edges_from(((2, 5), (6, 1)))
    GL = np.array([[0.1395, -0.0349, -0.0465, 0.0, 0.0, -0.0581], [-0.0349, 0.093, -0.0116, 0.0, -0.0465, 0.0], [-0.0465, -0.0116, 0.0698, 0.0, -0.0116, 0.0], [0.0, 0.0, 0.0, 0.2326, -0.1163, -0.1163], [0.0, -0.0465, -0.0116, -0.1163, 0.2326, -0.0581], [-0.0581, 0.0, 0.0, -0.1163, -0.0581, 0.2326]])
    L = nx.directed_combinatorial_laplacian_matrix(G, alpha=0.9, nodelist=sorted(G), walk_type='random')
    np.testing.assert_almost_equal(L, GL, decimal=3)
    GL = np.array([[0.0698, -0.0174, -0.0233, 0.0, 0.0, -0.0291], [-0.0174, 0.0465, -0.0058, 0.0, -0.0233, 0.0], [-0.0233, -0.0058, 0.0349, 0.0, -0.0058, 0.0], [0.0, 0.0, 0.0, 0.1163, -0.0581, -0.0581], [0.0, -0.0233, -0.0058, -0.0581, 0.1163, -0.0291], [-0.0291, 0.0, 0.0, -0.0581, -0.0291, 0.1163]])
    L = nx.directed_combinatorial_laplacian_matrix(G, alpha=0.9, nodelist=sorted(G), walk_type='lazy')
    np.testing.assert_almost_equal(L, GL, decimal=3)
    E = nx.DiGraph(margulis_gabber_galil_graph(2))
    L = nx.directed_combinatorial_laplacian_matrix(E)
    expected = np.array([[0.16666667, -0.08333333, -0.08333333, 0.0], [-0.08333333, 0.16666667, 0.0, -0.08333333], [-0.08333333, 0.0, 0.16666667, -0.08333333], [0.0, -0.08333333, -0.08333333, 0.16666667]])
    np.testing.assert_almost_equal(L, expected, decimal=6)
    with pytest.raises(nx.NetworkXError):
        nx.directed_combinatorial_laplacian_matrix(G, walk_type='pagerank', alpha=100)
    with pytest.raises(nx.NetworkXError):
        nx.directed_combinatorial_laplacian_matrix(G, walk_type='silly')