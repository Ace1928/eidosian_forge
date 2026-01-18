from unittest import mock
import netaddr
import testtools
from neutron_lib.api import converters
from neutron_lib import constants
from neutron_lib import exceptions as n_exc
from neutron_lib.tests import _base as base
from neutron_lib.tests import tools
def test_convert_to_float_none_value(self):
    self.assertIsNone(converters.convert_to_positive_float_or_none(None))