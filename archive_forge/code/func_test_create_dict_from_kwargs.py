from unittest import mock
from saharaclient.osc import utils
from saharaclient.tests.unit import base
def test_create_dict_from_kwargs(self):
    dict1 = utils.create_dict_from_kwargs(first='1', second=2)
    self.assertEqual({'first': '1', 'second': 2}, dict1)
    dict2 = utils.create_dict_from_kwargs(first='1', second=None)
    self.assertEqual({'first': '1'}, dict2)
    dict3 = utils.create_dict_from_kwargs(first='1', second=False)
    self.assertEqual({'first': '1', 'second': False}, dict3)