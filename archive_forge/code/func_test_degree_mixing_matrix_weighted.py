import pytest
import networkx as nx
from .base_test import BaseTestAttributeMixing, BaseTestDegreeMixing
def test_degree_mixing_matrix_weighted(self):
    a_result = np.array([[0.0, 1.0], [1.0, 6.0]])
    a = nx.degree_mixing_matrix(self.W, weight='weight', normalized=False)
    np.testing.assert_equal(a, a_result)
    a = nx.degree_mixing_matrix(self.W, weight='weight')
    np.testing.assert_equal(a, a_result / float(a_result.sum()))