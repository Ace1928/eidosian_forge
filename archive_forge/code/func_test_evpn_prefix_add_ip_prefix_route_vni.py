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
def test_evpn_prefix_add_ip_prefix_route_vni(self, mock_call):
    route_type = bgpspeaker.EVPN_IP_PREFIX_ROUTE
    route_dist = '65000:100'
    esi = 0
    ethernet_tag_id = 200
    ip_prefix = '192.168.0.0/24'
    gw_ip_addr = '172.16.0.1'
    vni = 500
    tunnel_type = bgpspeaker.TUNNEL_TYPE_VXLAN
    next_hop = '0.0.0.0'
    expected_kwargs = {'route_type': route_type, 'route_dist': route_dist, 'esi': esi, 'ethernet_tag_id': ethernet_tag_id, 'ip_prefix': ip_prefix, 'gw_ip_addr': gw_ip_addr, 'tunnel_type': tunnel_type, 'vni': vni, 'next_hop': next_hop}
    speaker = bgpspeaker.BGPSpeaker(65000, '10.0.0.1')
    speaker.evpn_prefix_add(route_type=route_type, route_dist=route_dist, esi=esi, ethernet_tag_id=ethernet_tag_id, ip_prefix=ip_prefix, gw_ip_addr=gw_ip_addr, tunnel_type=tunnel_type, vni=vni)
    mock_call.assert_called_with('evpn_prefix.add_local', **expected_kwargs)