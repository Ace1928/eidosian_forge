import pytest
import networkx as nx
from .base_test import BaseTestAttributeMixing, BaseTestDegreeMixing
def test_attribute_mixing_dict_undirected(self):
    d = nx.attribute_mixing_dict(self.G, 'fish')
    d_result = {'one': {'one': 2, 'red': 1}, 'two': {'two': 2, 'blue': 1}, 'red': {'one': 1}, 'blue': {'two': 1}}
    assert d == d_result