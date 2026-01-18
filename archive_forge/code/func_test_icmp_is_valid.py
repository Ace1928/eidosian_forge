from unittest import mock
import netaddr
import testtools
from neutron_lib.api import converters
from neutron_lib import constants
from neutron_lib import exceptions as n_exc
from neutron_lib.tests import _base as base
from neutron_lib.tests import tools
def test_icmp_is_valid(self):
    result = converters.convert_to_protocol(constants.PROTO_NAME_ICMP)
    self.assertEqual(constants.PROTO_NAME_ICMP, result)
    proto_num_str = str(constants.PROTO_NUM_ICMP)
    result = converters.convert_to_protocol(proto_num_str)
    self.assertEqual(proto_num_str, result)