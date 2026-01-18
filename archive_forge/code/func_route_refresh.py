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
def route_refresh(self, peer_ip=None, afi=None, safi=None):
    if not peer_ip:
        peer_ip = 'all'
    try:
        route_families = []
        if afi is None and safi is None:
            route_families.extend(SUPPORTED_GLOBAL_RF)
        else:
            route_family = RouteFamily(afi, safi)
            if route_family not in SUPPORTED_GLOBAL_RF:
                raise WrongParamError('Not supported address-family %s, %s' % (afi, safi))
            route_families.append(route_family)
        pm = CORE_MANAGER.get_core_service().peer_manager
        pm.make_route_refresh_request(peer_ip, *route_families)
    except Exception as e:
        LOG.error(traceback.format_exc())
        raise WrongParamError(str(e))
    return None