import pytest
import networkx as nx
from networkx.generators.degree_seq import havel_hakimi_graph
def test_bethe_hessian(self):
    """Bethe Hessian matrix"""
    H = np.array([[4, -2, 0], [-2, 5, -2], [0, -2, 4]])
    permutation = [2, 0, 1]
    np.testing.assert_equal(nx.bethe_hessian_matrix(self.P, r=2).todense(), H)
    np.testing.assert_equal(nx.bethe_hessian_matrix(self.P, r=2, nodelist=permutation).todense(), H[np.ix_(permutation, permutation)])
    np.testing.assert_equal(nx.bethe_hessian_matrix(self.G, r=1).todense(), nx.laplacian_matrix(self.G).todense())
    np.testing.assert_equal(nx.bethe_hessian_matrix(self.G).todense(), nx.bethe_hessian_matrix(self.G, r=1.25).todense())