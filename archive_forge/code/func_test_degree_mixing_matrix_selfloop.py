import pytest
import networkx as nx
from .base_test import BaseTestAttributeMixing, BaseTestDegreeMixing
def test_degree_mixing_matrix_selfloop(self):
    a_result = np.array([[2]])
    a = nx.degree_mixing_matrix(self.S, normalized=False)
    np.testing.assert_equal(a, a_result)
    a = nx.degree_mixing_matrix(self.S)
    np.testing.assert_equal(a, a_result / a_result.sum())