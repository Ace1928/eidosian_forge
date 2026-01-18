import pytest
import networkx as nx
from .base_test import BaseTestAttributeMixing, BaseTestDegreeMixing
def test_attribute_mixing_matrix_negative(self):
    mapping = {-2: 0, -3: 1, -4: 2}
    a_result = np.array([[4.0, 1.0, 1.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    a = nx.attribute_mixing_matrix(self.N, 'margin', mapping=mapping, normalized=False)
    np.testing.assert_equal(a, a_result)
    a = nx.attribute_mixing_matrix(self.N, 'margin', mapping=mapping)
    np.testing.assert_equal(a, a_result / float(a_result.sum()))