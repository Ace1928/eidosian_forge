import pytest
import networkx as nx
from networkx.generators.degree_seq import havel_hakimi_graph
from networkx.generators.expanders import margulis_gabber_galil_graph
def test_directed_laplacian():
    """Directed Laplacian"""
    G = nx.DiGraph()
    G.add_edges_from(((1, 2), (1, 3), (3, 1), (3, 2), (3, 5), (4, 5), (4, 6), (5, 4), (5, 6), (6, 4)))
    GL = np.array([[0.9833, -0.2941, -0.3882, -0.0291, -0.0231, -0.0261], [-0.2941, 0.8333, -0.2339, -0.0536, -0.0589, -0.0554], [-0.3882, -0.2339, 0.9833, -0.0278, -0.0896, -0.0251], [-0.0291, -0.0536, -0.0278, 0.9833, -0.4878, -0.6675], [-0.0231, -0.0589, -0.0896, -0.4878, 0.9833, -0.2078], [-0.0261, -0.0554, -0.0251, -0.6675, -0.2078, 0.9833]])
    L = nx.directed_laplacian_matrix(G, alpha=0.9, nodelist=sorted(G))
    np.testing.assert_almost_equal(L, GL, decimal=3)
    G.add_edges_from(((2, 5), (6, 1)))
    GL = np.array([[1.0, -0.3062, -0.4714, 0.0, 0.0, -0.3227], [-0.3062, 1.0, -0.1443, 0.0, -0.3162, 0.0], [-0.4714, -0.1443, 1.0, 0.0, -0.0913, 0.0], [0.0, 0.0, 0.0, 1.0, -0.5, -0.5], [0.0, -0.3162, -0.0913, -0.5, 1.0, -0.25], [-0.3227, 0.0, 0.0, -0.5, -0.25, 1.0]])
    L = nx.directed_laplacian_matrix(G, alpha=0.9, nodelist=sorted(G), walk_type='random')
    np.testing.assert_almost_equal(L, GL, decimal=3)
    GL = np.array([[0.5, -0.1531, -0.2357, 0.0, 0.0, -0.1614], [-0.1531, 0.5, -0.0722, 0.0, -0.1581, 0.0], [-0.2357, -0.0722, 0.5, 0.0, -0.0456, 0.0], [0.0, 0.0, 0.0, 0.5, -0.25, -0.25], [0.0, -0.1581, -0.0456, -0.25, 0.5, -0.125], [-0.1614, 0.0, 0.0, -0.25, -0.125, 0.5]])
    L = nx.directed_laplacian_matrix(G, alpha=0.9, nodelist=sorted(G), walk_type='lazy')
    np.testing.assert_almost_equal(L, GL, decimal=3)
    G = nx.DiGraph()
    G.add_edges_from(((1, 2), (2, 4), (4, 1), (1, 3), (3, 4)))
    GL = np.array([[0.5, -0.176, -0.176, -0.25], [-0.176, 0.5, 0.0, -0.176], [-0.176, 0.0, 0.5, -0.176], [-0.25, -0.176, -0.176, 0.5]])
    L = nx.directed_laplacian_matrix(G, alpha=0.9, nodelist=sorted(G))
    np.testing.assert_almost_equal(L, GL, decimal=3)