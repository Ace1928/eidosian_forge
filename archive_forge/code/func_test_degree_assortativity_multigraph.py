import pytest
import networkx as nx
from networkx.algorithms.assortativity.correlation import attribute_ac
from .base_test import BaseTestAttributeMixing, BaseTestDegreeMixing
def test_degree_assortativity_multigraph(self):
    r = nx.degree_assortativity_coefficient(self.M)
    np.testing.assert_almost_equal(r, -1.0 / 7.0, decimal=4)