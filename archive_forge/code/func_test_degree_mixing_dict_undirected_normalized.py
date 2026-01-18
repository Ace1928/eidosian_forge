import pytest
import networkx as nx
from .base_test import BaseTestAttributeMixing, BaseTestDegreeMixing
def test_degree_mixing_dict_undirected_normalized(self):
    d = nx.degree_mixing_dict(self.P4, normalized=True)
    d_result = {1: {2: 1.0 / 3}, 2: {1: 1.0 / 3, 2: 1.0 / 3}}
    assert d == d_result