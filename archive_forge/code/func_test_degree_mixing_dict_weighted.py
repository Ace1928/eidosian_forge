import pytest
import networkx as nx
from .base_test import BaseTestAttributeMixing, BaseTestDegreeMixing
def test_degree_mixing_dict_weighted(self):
    d = nx.degree_mixing_dict(self.W, weight='weight')
    d_result = {0.5: {1.5: 1}, 1.5: {1.5: 6, 0.5: 1}}
    assert d == d_result