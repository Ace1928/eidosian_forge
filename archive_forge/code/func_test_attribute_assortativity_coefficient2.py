import pytest
import networkx as nx
from networkx.algorithms.assortativity.correlation import attribute_ac
from .base_test import BaseTestAttributeMixing, BaseTestDegreeMixing
def test_attribute_assortativity_coefficient2(self):
    a = np.array([[0.18, 0.02, 0.01, 0.03], [0.02, 0.2, 0.03, 0.02], [0.01, 0.03, 0.16, 0.01], [0.03, 0.02, 0.01, 0.22]])
    r = attribute_ac(a)
    np.testing.assert_almost_equal(r, 0.68, decimal=2)