from unittest import mock
import netaddr
import testtools
from neutron_lib.api import converters
from neutron_lib import constants
from neutron_lib import exceptions as n_exc
from neutron_lib.tests import _base as base
from neutron_lib.tests import tools
def test_convert_to_int_int(self):
    self.assertEqual(-1, converters.convert_to_int(-1))
    self.assertEqual(0, converters.convert_to_int(0))
    self.assertEqual(1, converters.convert_to_int(1))