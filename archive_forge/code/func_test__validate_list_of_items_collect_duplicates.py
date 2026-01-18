import string
from unittest import mock
import netaddr
from neutron_lib._i18n import _
from neutron_lib.api import converters
from neutron_lib.api.definitions import extra_dhcp_opt
from neutron_lib.api import validators
from neutron_lib import constants
from neutron_lib import exceptions as n_exc
from neutron_lib import fixture
from neutron_lib.plugins import directory
from neutron_lib.tests import _base as base
def test__validate_list_of_items_collect_duplicates(self):
    items = ['a', 'b', 'duplicate_1', 'duplicate_2', 'duplicate_1', 'duplicate_2', 'duplicate_2', 'c']
    msg = validators._validate_list_of_items(mock.Mock(), items)
    error = "Duplicate items in the list: '%s'" % 'duplicate_1, duplicate_2'
    self.assertEqual(error, msg)
    items = [['a', 'b'], ['c', 'd'], ['a', 'b']]
    msg = validators._validate_list_of_items(mock.Mock(), items)
    error = "Duplicate items in the list: '%s'" % str(['a', 'b'])
    self.assertEqual(error, msg)
    items = [{'a': 'b'}, {'c': 'd'}, {'a': 'b'}]
    msg = validators._validate_list_of_items(mock.Mock(), items)
    error = "Duplicate items in the list: '%s'" % str({'a': 'b'})
    self.assertEqual(error, msg)