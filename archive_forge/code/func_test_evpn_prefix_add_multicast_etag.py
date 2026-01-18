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
def test_evpn_prefix_add_multicast_etag(self, mock_call):
    route_type = bgpspeaker.EVPN_MULTICAST_ETAG_ROUTE
    route_dist = '65000:100'
    esi = 0
    ethernet_tag_id = 200
    mac_addr = 'aa:bb:cc:dd:ee:ff'
    ip_addr = '192.168.0.1'
    next_hop = '10.0.0.1'
    expected_kwargs = {'route_type': route_type, 'route_dist': route_dist, 'ethernet_tag_id': ethernet_tag_id, 'ip_addr': ip_addr, 'next_hop': next_hop}
    speaker = bgpspeaker.BGPSpeaker(65000, '10.0.0.1')
    speaker.evpn_prefix_add(route_type=route_type, route_dist=route_dist, esi=esi, ethernet_tag_id=ethernet_tag_id, mac_addr=mac_addr, ip_addr=ip_addr, next_hop=next_hop)
    mock_call.assert_called_with('evpn_prefix.add_local', **expected_kwargs)