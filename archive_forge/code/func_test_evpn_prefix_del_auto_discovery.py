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
def test_evpn_prefix_del_auto_discovery(self, mock_call):
    route_type = bgpspeaker.EVPN_ETH_AUTO_DISCOVERY
    route_dist = '65000:100'
    esi = {'type': ESI_TYPE_LACP, 'mac_addr': 'aa:bb:cc:dd:ee:ff', 'port_key': 100}
    ethernet_tag_id = EVPN_MAX_ET
    expected_kwargs = {'route_type': route_type, 'route_dist': route_dist, 'esi': esi, 'ethernet_tag_id': ethernet_tag_id}
    speaker = bgpspeaker.BGPSpeaker(65000, '10.0.0.1')
    speaker.evpn_prefix_del(route_type=route_type, route_dist=route_dist, esi=esi, ethernet_tag_id=ethernet_tag_id)
    mock_call.assert_called_with('evpn_prefix.delete_local', **expected_kwargs)