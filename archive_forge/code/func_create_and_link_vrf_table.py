import logging
from collections import OrderedDict
import netaddr
from os_ken.services.protocols.bgp.base import SUPPORTED_GLOBAL_RF
from os_ken.services.protocols.bgp.info_base.rtc import RtcTable
from os_ken.services.protocols.bgp.info_base.ipv4 import Ipv4Path
from os_ken.services.protocols.bgp.info_base.ipv4 import Ipv4Table
from os_ken.services.protocols.bgp.info_base.ipv6 import Ipv6Path
from os_ken.services.protocols.bgp.info_base.ipv6 import Ipv6Table
from os_ken.services.protocols.bgp.info_base.vpnv4 import Vpnv4Table
from os_ken.services.protocols.bgp.info_base.vpnv6 import Vpnv6Table
from os_ken.services.protocols.bgp.info_base.vrf4 import Vrf4Table
from os_ken.services.protocols.bgp.info_base.vrf6 import Vrf6Table
from os_ken.services.protocols.bgp.info_base.vrfevpn import VrfEvpnTable
from os_ken.services.protocols.bgp.info_base.evpn import EvpnTable
from os_ken.services.protocols.bgp.info_base.ipv4fs import IPv4FlowSpecPath
from os_ken.services.protocols.bgp.info_base.ipv4fs import IPv4FlowSpecTable
from os_ken.services.protocols.bgp.info_base.vpnv4fs import VPNv4FlowSpecTable
from os_ken.services.protocols.bgp.info_base.vrf4fs import Vrf4FlowSpecTable
from os_ken.services.protocols.bgp.info_base.ipv6fs import IPv6FlowSpecPath
from os_ken.services.protocols.bgp.info_base.ipv6fs import IPv6FlowSpecTable
from os_ken.services.protocols.bgp.info_base.vpnv6fs import VPNv6FlowSpecTable
from os_ken.services.protocols.bgp.info_base.vrf6fs import Vrf6FlowSpecTable
from os_ken.services.protocols.bgp.info_base.l2vpnfs import L2VPNFlowSpecTable
from os_ken.services.protocols.bgp.info_base.vrfl2vpnfs import L2vpnFlowSpecPath
from os_ken.services.protocols.bgp.info_base.vrfl2vpnfs import L2vpnFlowSpecTable
from os_ken.services.protocols.bgp.rtconf.vrfs import VRF_RF_IPV4
from os_ken.services.protocols.bgp.rtconf.vrfs import VRF_RF_IPV6
from os_ken.services.protocols.bgp.rtconf.vrfs import VRF_RF_L2_EVPN
from os_ken.services.protocols.bgp.rtconf.vrfs import VRF_RF_IPV4_FLOWSPEC
from os_ken.services.protocols.bgp.rtconf.vrfs import VRF_RF_IPV6_FLOWSPEC
from os_ken.services.protocols.bgp.rtconf.vrfs import VRF_RF_L2VPN_FLOWSPEC
from os_ken.services.protocols.bgp.rtconf.vrfs import SUPPORTED_VRF_RF
from os_ken.services.protocols.bgp.utils.bgp import create_v4flowspec_actions
from os_ken.services.protocols.bgp.utils.bgp import create_v6flowspec_actions
from os_ken.services.protocols.bgp.utils.bgp import create_l2vpnflowspec_actions
from os_ken.lib import type_desc
from os_ken.lib import ip
from os_ken.lib.packet.bgp import RF_IPv4_UC
from os_ken.lib.packet.bgp import RF_IPv6_UC
from os_ken.lib.packet.bgp import RF_IPv4_VPN
from os_ken.lib.packet.bgp import RF_IPv6_VPN
from os_ken.lib.packet.bgp import RF_L2_EVPN
from os_ken.lib.packet.bgp import RF_IPv4_FLOWSPEC
from os_ken.lib.packet.bgp import RF_IPv6_FLOWSPEC
from os_ken.lib.packet.bgp import RF_VPNv4_FLOWSPEC
from os_ken.lib.packet.bgp import RF_VPNv6_FLOWSPEC
from os_ken.lib.packet.bgp import RF_L2VPN_FLOWSPEC
from os_ken.lib.packet.bgp import RF_RTC_UC
from os_ken.lib.packet.bgp import BGPPathAttributeOrigin
from os_ken.lib.packet.bgp import BGPPathAttributeAsPath
from os_ken.lib.packet.bgp import BGPPathAttributeExtendedCommunities
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_ORIGIN
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_AS_PATH
from os_ken.lib.packet.bgp import BGP_ATTR_ORIGIN_IGP
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_EXTENDED_COMMUNITIES
from os_ken.lib.packet.bgp import EvpnEsi
from os_ken.lib.packet.bgp import EvpnArbitraryEsi
from os_ken.lib.packet.bgp import EvpnNLRI
from os_ken.lib.packet.bgp import EvpnMacIPAdvertisementNLRI
from os_ken.lib.packet.bgp import EvpnInclusiveMulticastEthernetTagNLRI
from os_ken.lib.packet.bgp import IPAddrPrefix
from os_ken.lib.packet.bgp import IP6AddrPrefix
from os_ken.lib.packet.bgp import FlowSpecIPv4NLRI
from os_ken.lib.packet.bgp import FlowSpecIPv6NLRI
from os_ken.lib.packet.bgp import FlowSpecL2VPNNLRI
from os_ken.services.protocols.bgp.utils.validation import is_valid_ipv4
from os_ken.services.protocols.bgp.utils.validation import is_valid_ipv4_prefix
from os_ken.services.protocols.bgp.utils.validation import is_valid_ipv6
from os_ken.services.protocols.bgp.utils.validation import is_valid_ipv6_prefix
def create_and_link_vrf_table(self, vrf_conf):
    """Factory method to create VRF table for given `vrf_conf`.

        Adds mapping to this table with appropriate scope. Also, adds mapping
        for import RT of this VRF to created table to facilitate
        importing/installing of paths from global tables.
        Returns created table.
        """
    route_family = vrf_conf.route_family
    if route_family == VRF_RF_IPV4:
        vrf_table = Vrf4Table
    elif route_family == VRF_RF_IPV6:
        vrf_table = Vrf6Table
    elif route_family == VRF_RF_L2_EVPN:
        vrf_table = VrfEvpnTable
    elif route_family == VRF_RF_IPV4_FLOWSPEC:
        vrf_table = Vrf4FlowSpecTable
    elif route_family == VRF_RF_IPV6_FLOWSPEC:
        vrf_table = Vrf6FlowSpecTable
    elif route_family == VRF_RF_L2VPN_FLOWSPEC:
        vrf_table = L2vpnFlowSpecTable
    else:
        raise ValueError('Unsupported route family for VRF: %s' % route_family)
    vrf_table = vrf_table(vrf_conf, self._core_service, self._signal_bus)
    table_id = (vrf_conf.route_dist, route_family)
    self._tables[table_id] = vrf_table
    assert vrf_table is not None
    LOG.debug('Added new VrfTable with route_dist:%s and route_family:%s', vrf_conf.route_dist, route_family)
    import_rts = vrf_conf.import_rts
    if import_rts:
        self._link_vrf_table(vrf_table, import_rts)
    return vrf_table