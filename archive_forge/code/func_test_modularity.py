import pytest
import networkx as nx
from networkx.generators.degree_seq import havel_hakimi_graph
def test_modularity(self):
    """Modularity matrix"""
    B = np.array([[-1.125, 0.25, 0.25, 0.625, 0.0], [0.25, -0.5, 0.5, -0.25, 0.0], [0.25, 0.5, -0.5, -0.25, 0.0], [0.625, -0.25, -0.25, -0.125, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]])
    permutation = [4, 0, 1, 2, 3]
    np.testing.assert_equal(nx.modularity_matrix(self.G), B)
    np.testing.assert_equal(nx.modularity_matrix(self.G, nodelist=permutation), B[np.ix_(permutation, permutation)])