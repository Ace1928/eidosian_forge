import pytest
import networkx as nx
from .base_test import BaseTestAttributeMixing, BaseTestDegreeMixing
def test_attribute_mixing_matrix_float(self):
    mapping = {0.5: 1, 1.5: 0}
    a_result = np.array([[6.0, 1.0], [1.0, 0.0]])
    a = nx.attribute_mixing_matrix(self.F, 'margin', mapping=mapping, normalized=False)
    np.testing.assert_equal(a, a_result)
    a = nx.attribute_mixing_matrix(self.F, 'margin', mapping=mapping)
    np.testing.assert_equal(a, a_result / a_result.sum())