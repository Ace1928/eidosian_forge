from unittest import mock
import netaddr
import testtools
from neutron_lib.api import converters
from neutron_lib import constants
from neutron_lib import exceptions as n_exc
from neutron_lib.tests import _base as base
from neutron_lib.tests import tools
def test_convert_none_to_empty_list_value(self):
    values = ['1', 3, [], [1], {}, {'a': 3}]
    for value in values:
        self.assertEqual(value, converters.convert_none_to_empty_list(value))