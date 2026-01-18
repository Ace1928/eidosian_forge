import pytest
import networkx as nx
from networkx.algorithms.assortativity.correlation import attribute_ac
from .base_test import BaseTestAttributeMixing, BaseTestDegreeMixing
def test_attribute_assortativity_mixed(self):
    r = nx.numeric_assortativity_coefficient(self.K, 'margin')
    np.testing.assert_almost_equal(r, 0.434, decimal=4)