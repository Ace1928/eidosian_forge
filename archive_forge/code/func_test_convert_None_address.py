from unittest import mock
import netaddr
import testtools
from neutron_lib.api import converters
from neutron_lib import constants
from neutron_lib import exceptions as n_exc
from neutron_lib.tests import _base as base
from neutron_lib.tests import tools
def test_convert_None_address(self):
    result = converters.convert_ip_to_canonical_format(None)
    self.assertIsNone(result)