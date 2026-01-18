import pytest
import networkx as nx
from networkx.algorithms.assortativity.correlation import attribute_ac
from .base_test import BaseTestAttributeMixing, BaseTestDegreeMixing
def test_attribute_assortativity_multigraph(self):
    r = nx.attribute_assortativity_coefficient(self.M, 'fish')
    assert r == 1.0