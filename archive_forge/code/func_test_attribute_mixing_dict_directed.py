import pytest
import networkx as nx
from .base_test import BaseTestAttributeMixing, BaseTestDegreeMixing
def test_attribute_mixing_dict_directed(self):
    d = nx.attribute_mixing_dict(self.D, 'fish')
    d_result = {'one': {'one': 1, 'red': 1}, 'two': {'two': 1, 'blue': 1}, 'red': {}, 'blue': {}}
    assert d == d_result