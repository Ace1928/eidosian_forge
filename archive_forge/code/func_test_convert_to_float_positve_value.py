from unittest import mock
import netaddr
import testtools
from neutron_lib.api import converters
from neutron_lib import constants
from neutron_lib import exceptions as n_exc
from neutron_lib.tests import _base as base
from neutron_lib.tests import tools
def test_convert_to_float_positve_value(self):
    self.assertEqual(1.111, converters.convert_to_positive_float_or_none(1.111))
    self.assertEqual(1, converters.convert_to_positive_float_or_none(1))
    self.assertEqual(0, converters.convert_to_positive_float_or_none(0))