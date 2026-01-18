import pytest
import networkx as nx
from networkx.algorithms.assortativity.correlation import attribute_ac
from .base_test import BaseTestAttributeMixing, BaseTestDegreeMixing
def test_degree_pearson_assortativity_directed(self):
    r = nx.degree_pearson_correlation_coefficient(self.D)
    np.testing.assert_almost_equal(r, -0.57735, decimal=4)