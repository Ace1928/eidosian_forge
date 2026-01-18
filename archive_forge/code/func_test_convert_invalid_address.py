from unittest import mock
import netaddr
import testtools
from neutron_lib.api import converters
from neutron_lib import constants
from neutron_lib import exceptions as n_exc
from neutron_lib.tests import _base as base
from neutron_lib.tests import tools
def test_convert_invalid_address(self):
    result = converters.convert_ip_to_canonical_format('on')
    self.assertEqual('on', result)
    result = converters.convert_ip_to_canonical_format('192.168.1.1/32')
    self.assertEqual('192.168.1.1/32', result)
    result = converters.convert_ip_to_canonical_format('2001:db8:0:1:1:1:1:1/128')
    self.assertEqual('2001:db8:0:1:1:1:1:1/128', result)