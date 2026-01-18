import pytest
import networkx as nx
from networkx.algorithms.assortativity.correlation import attribute_ac
from .base_test import BaseTestAttributeMixing, BaseTestDegreeMixing
def test_degree_assortativity_directed2(self):
    """Test degree assortativity for a directed graph where the set of
        in/out degree does not equal the total degree."""
    r = nx.degree_assortativity_coefficient(self.D2)
    np.testing.assert_almost_equal(r, 0.14852, decimal=4)