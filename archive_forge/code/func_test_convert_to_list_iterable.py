from unittest import mock
import netaddr
import testtools
from neutron_lib.api import converters
from neutron_lib import constants
from neutron_lib import exceptions as n_exc
from neutron_lib.tests import _base as base
from neutron_lib.tests import tools
def test_convert_to_list_iterable(self):
    for item in ([None], [1, 2, 3], (1, 2, 3), set([1, 2, 3]), ['foo']):
        self.assertEqual(list(item), converters.convert_to_list(item))