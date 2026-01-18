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
def import_single_vpn_path_to_all_vrfs(self, vpn_path, path_rts=None):
    """Imports *vpn_path* to qualifying VRF tables.

        Import RTs of VRF table is matched with RTs from *vpn4_path* and if we
        have any common RTs we import the path into VRF.
        """
    LOG.debug('Importing path %s to qualifying VRFs', vpn_path)
    if not path_rts:
        LOG.info('Encountered a path with no RTs: %s', vpn_path)
        return
    interested_tables = set()
    if vpn_path.route_family == RF_IPv4_VPN:
        route_family = RF_IPv4_UC
    elif vpn_path.route_family == RF_IPv6_VPN:
        route_family = RF_IPv6_UC
    elif vpn_path.route_family == RF_L2_EVPN:
        route_family = RF_L2_EVPN
    elif vpn_path.route_family == RF_VPNv4_FLOWSPEC:
        route_family = RF_IPv4_FLOWSPEC
    elif vpn_path.route_family == RF_VPNv6_FLOWSPEC:
        route_family = RF_IPv6_FLOWSPEC
    elif vpn_path.route_family == RF_L2VPN_FLOWSPEC:
        route_family = RF_L2VPN_FLOWSPEC
    else:
        raise ValueError('Unsupported route family for VRF: %s' % vpn_path.route_family)
    for rt in path_rts:
        rt_rf_id = rt + ':' + str(route_family)
        vrf_rt_tables = self._tables_for_rt.get(rt_rf_id)
        if vrf_rt_tables:
            interested_tables.update(vrf_rt_tables)
    if interested_tables:
        route_dist = vpn_path.nlri.route_dist
        for vrf_table in interested_tables:
            if vpn_path.source is not None or route_dist != vrf_table.vrf_conf.route_dist:
                update_vrf_dest = vrf_table.import_vpn_path(vpn_path)
                if update_vrf_dest is not None:
                    self._signal_bus.dest_changed(update_vrf_dest)
    else:
        LOG.debug('No VRF table found that imports RTs: %s', path_rts)