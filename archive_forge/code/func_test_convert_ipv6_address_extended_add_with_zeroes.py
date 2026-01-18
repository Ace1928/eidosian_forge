from unittest import mock
import netaddr
import testtools
from neutron_lib.api import converters
from neutron_lib import constants
from neutron_lib import exceptions as n_exc
from neutron_lib.tests import _base as base
from neutron_lib.tests import tools
def test_convert_ipv6_address_extended_add_with_zeroes(self):
    result = converters.convert_ip_to_canonical_format('2001:0db8:0:0:0:0:0:0001')
    self.assertEqual('2001:db8::1', result)