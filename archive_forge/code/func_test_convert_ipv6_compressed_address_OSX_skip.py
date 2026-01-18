from unittest import mock
import netaddr
import testtools
from neutron_lib.api import converters
from neutron_lib import constants
from neutron_lib import exceptions as n_exc
from neutron_lib.tests import _base as base
from neutron_lib.tests import tools
@testtools.skipIf(tools.is_bsd(), 'bug/1484837')
def test_convert_ipv6_compressed_address_OSX_skip(self):
    result = converters.convert_ip_to_canonical_format('2001:db8:0:1:1:1:1:1')
    self.assertEqual('2001:db8:0:1:1:1:1:1', result)