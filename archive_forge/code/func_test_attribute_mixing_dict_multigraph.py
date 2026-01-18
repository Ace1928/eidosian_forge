import pytest
import networkx as nx
from .base_test import BaseTestAttributeMixing, BaseTestDegreeMixing
def test_attribute_mixing_dict_multigraph(self):
    d = nx.attribute_mixing_dict(self.M, 'fish')
    d_result = {'one': {'one': 4}, 'two': {'two': 2}}
    assert d == d_result