import logging
import traceback
from os_ken.lib.packet.bgp import RouteFamily
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
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_ORIGIN
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_AS_PATH
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_MULTI_EXIT_DISC
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_LOCAL_PREF
from os_ken.lib.packet.bgp import BGP_ATTR_ORIGIN_IGP
from os_ken.lib.packet.bgp import BGP_ATTR_ORIGIN_EGP
from os_ken.lib.packet.bgp import BGP_ATTR_ORIGIN_INCOMPLETE
from os_ken.services.protocols.bgp.base import add_bgp_error_metadata
from os_ken.services.protocols.bgp.base import BGPSException
from os_ken.services.protocols.bgp.base import SUPPORTED_GLOBAL_RF
from os_ken.services.protocols.bgp.core_manager import CORE_MANAGER
def get_single_rib_routes(self, addr_family):
    rfs = {'ipv4': RF_IPv4_UC, 'ipv6': RF_IPv6_UC, 'vpnv4': RF_IPv4_VPN, 'vpnv6': RF_IPv6_VPN, 'evpn': RF_L2_EVPN, 'ipv4fs': RF_IPv4_FLOWSPEC, 'ipv6fs': RF_IPv6_FLOWSPEC, 'vpnv4fs': RF_VPNv4_FLOWSPEC, 'vpnv6fs': RF_VPNv6_FLOWSPEC, 'l2vpnfs': RF_L2VPN_FLOWSPEC, 'rtfilter': RF_RTC_UC}
    if addr_family not in rfs:
        raise WrongParamError('Unknown or unsupported family: %s' % addr_family)
    rf = rfs.get(addr_family)
    table_manager = self.get_core_service().table_manager
    gtable = table_manager.get_global_table_by_route_family(rf)
    if gtable is not None:
        return [self._dst_to_dict(dst) for dst in sorted(gtable.values())]
    else:
        return []