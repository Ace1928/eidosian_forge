import logging
import unittest
from unittest import mock
from os_ken.services.protocols.bgp import bgpspeaker
from os_ken.services.protocols.bgp.bgpspeaker import EVPN_MAX_ET
from os_ken.services.protocols.bgp.bgpspeaker import ESI_TYPE_LACP
from os_ken.services.protocols.bgp.api.prefix import ESI_TYPE_L2_BRIDGE
from os_ken.services.protocols.bgp.bgpspeaker import ESI_TYPE_MAC_BASED
from os_ken.services.protocols.bgp.api.prefix import REDUNDANCY_MODE_ALL_ACTIVE
from os_ken.services.protocols.bgp.api.prefix import REDUNDANCY_MODE_SINGLE_ACTIVE
@mock.patch('os_ken.services.protocols.bgp.bgpspeaker.BGPSpeaker.__init__', mock.MagicMock(return_value=None))
@mock.patch('os_ken.services.protocols.bgp.bgpspeaker.call')
def test_flowspec_prefix_del_ipv6(self, mock_call):
    flowspec_family = bgpspeaker.FLOWSPEC_FAMILY_IPV6
    rules = {'dst_prefix': '2001::3/128/32'}
    expected_kwargs = {'flowspec_family': flowspec_family, 'rules': rules}
    speaker = bgpspeaker.BGPSpeaker(65000, '10.0.0.1')
    speaker.flowspec_prefix_del(flowspec_family=flowspec_family, rules=rules)
    mock_call.assert_called_with('flowspec.del', **expected_kwargs)