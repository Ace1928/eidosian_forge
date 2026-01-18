import pytest
import networkx as nx
from networkx.algorithms.assortativity.correlation import attribute_ac
from .base_test import BaseTestAttributeMixing, BaseTestDegreeMixing
def test_degree_assortativity_weighted(self):
    r = nx.degree_assortativity_coefficient(self.W, weight='weight')
    np.testing.assert_almost_equal(r, -0.1429, decimal=4)