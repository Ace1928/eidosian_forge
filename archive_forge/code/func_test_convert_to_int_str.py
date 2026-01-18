from unittest import mock
import netaddr
import testtools
from neutron_lib.api import converters
from neutron_lib import constants
from neutron_lib import exceptions as n_exc
from neutron_lib.tests import _base as base
from neutron_lib.tests import tools
def test_convert_to_int_str(self):
    self.assertEqual(4, converters.convert_to_int('4'))
    self.assertEqual(6, converters.convert_to_int('6'))
    self.assertRaises(n_exc.InvalidInput, converters.convert_to_int, 'garbage')