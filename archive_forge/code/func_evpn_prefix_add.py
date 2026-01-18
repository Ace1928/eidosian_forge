import netaddr
from os_ken.lib import hub
from os_ken.lib import ip
from os_ken.lib.packet.bgp import (
from os_ken.services.protocols.bgp.core_manager import CORE_MANAGER
from os_ken.services.protocols.bgp.signals.emit import BgpSignalBus
from os_ken.services.protocols.bgp.api.base import call
from os_ken.services.protocols.bgp.api.base import PREFIX
from os_ken.services.protocols.bgp.api.base import EVPN_ROUTE_TYPE
from os_ken.services.protocols.bgp.api.base import EVPN_ESI
from os_ken.services.protocols.bgp.api.base import EVPN_ETHERNET_TAG_ID
from os_ken.services.protocols.bgp.api.base import REDUNDANCY_MODE
from os_ken.services.protocols.bgp.api.base import IP_ADDR
from os_ken.services.protocols.bgp.api.base import MAC_ADDR
from os_ken.services.protocols.bgp.api.base import NEXT_HOP
from os_ken.services.protocols.bgp.api.base import IP_PREFIX
from os_ken.services.protocols.bgp.api.base import GW_IP_ADDR
from os_ken.services.protocols.bgp.api.base import ROUTE_DISTINGUISHER
from os_ken.services.protocols.bgp.api.base import ROUTE_FAMILY
from os_ken.services.protocols.bgp.api.base import EVPN_VNI
from os_ken.services.protocols.bgp.api.base import TUNNEL_TYPE
from os_ken.services.protocols.bgp.api.base import PMSI_TUNNEL_TYPE
from os_ken.services.protocols.bgp.api.base import MAC_MOBILITY
from os_ken.services.protocols.bgp.api.base import TUNNEL_ENDPOINT_IP
from os_ken.services.protocols.bgp.api.prefix import EVPN_MAX_ET
from os_ken.services.protocols.bgp.api.prefix import ESI_TYPE_LACP
from os_ken.services.protocols.bgp.api.prefix import ESI_TYPE_L2_BRIDGE
from os_ken.services.protocols.bgp.api.prefix import ESI_TYPE_MAC_BASED
from os_ken.services.protocols.bgp.api.prefix import EVPN_ETH_AUTO_DISCOVERY
from os_ken.services.protocols.bgp.api.prefix import EVPN_MAC_IP_ADV_ROUTE
from os_ken.services.protocols.bgp.api.prefix import EVPN_MULTICAST_ETAG_ROUTE
from os_ken.services.protocols.bgp.api.prefix import EVPN_ETH_SEGMENT
from os_ken.services.protocols.bgp.api.prefix import EVPN_IP_PREFIX_ROUTE
from os_ken.services.protocols.bgp.api.prefix import REDUNDANCY_MODE_ALL_ACTIVE
from os_ken.services.protocols.bgp.api.prefix import REDUNDANCY_MODE_SINGLE_ACTIVE
from os_ken.services.protocols.bgp.api.prefix import TUNNEL_TYPE_VXLAN
from os_ken.services.protocols.bgp.api.prefix import TUNNEL_TYPE_NVGRE
from os_ken.services.protocols.bgp.api.prefix import (
from os_ken.services.protocols.bgp.api.prefix import (
from os_ken.services.protocols.bgp.model import ReceivedRoute
from os_ken.services.protocols.bgp.rtconf.common import LOCAL_AS
from os_ken.services.protocols.bgp.rtconf.common import ROUTER_ID
from os_ken.services.protocols.bgp.rtconf.common import CLUSTER_ID
from os_ken.services.protocols.bgp.rtconf.common import BGP_SERVER_HOSTS
from os_ken.services.protocols.bgp.rtconf.common import BGP_SERVER_PORT
from os_ken.services.protocols.bgp.rtconf.common import DEFAULT_BGP_SERVER_HOSTS
from os_ken.services.protocols.bgp.rtconf.common import DEFAULT_BGP_SERVER_PORT
from os_ken.services.protocols.bgp.rtconf.common import (
from os_ken.services.protocols.bgp.rtconf.common import DEFAULT_LABEL_RANGE
from os_ken.services.protocols.bgp.rtconf.common import REFRESH_MAX_EOR_TIME
from os_ken.services.protocols.bgp.rtconf.common import REFRESH_STALEPATH_TIME
from os_ken.services.protocols.bgp.rtconf.common import LABEL_RANGE
from os_ken.services.protocols.bgp.rtconf.common import ALLOW_LOCAL_AS_IN_COUNT
from os_ken.services.protocols.bgp.rtconf.common import LOCAL_PREF
from os_ken.services.protocols.bgp.rtconf.common import DEFAULT_LOCAL_PREF
from os_ken.services.protocols.bgp.rtconf import neighbors
from os_ken.services.protocols.bgp.rtconf import vrfs
from os_ken.services.protocols.bgp.rtconf.base import CAP_MBGP_IPV4
from os_ken.services.protocols.bgp.rtconf.base import CAP_MBGP_IPV6
from os_ken.services.protocols.bgp.rtconf.base import CAP_MBGP_VPNV4
from os_ken.services.protocols.bgp.rtconf.base import CAP_MBGP_VPNV6
from os_ken.services.protocols.bgp.rtconf.base import CAP_MBGP_EVPN
from os_ken.services.protocols.bgp.rtconf.base import CAP_MBGP_IPV4FS
from os_ken.services.protocols.bgp.rtconf.base import CAP_MBGP_IPV6FS
from os_ken.services.protocols.bgp.rtconf.base import CAP_MBGP_VPNV4FS
from os_ken.services.protocols.bgp.rtconf.base import CAP_MBGP_VPNV6FS
from os_ken.services.protocols.bgp.rtconf.base import CAP_MBGP_L2VPNFS
from os_ken.services.protocols.bgp.rtconf.base import CAP_ENHANCED_REFRESH
from os_ken.services.protocols.bgp.rtconf.base import CAP_FOUR_OCTET_AS_NUMBER
from os_ken.services.protocols.bgp.rtconf.base import MULTI_EXIT_DISC
from os_ken.services.protocols.bgp.rtconf.base import SITE_OF_ORIGINS
from os_ken.services.protocols.bgp.rtconf.neighbors import (
from os_ken.services.protocols.bgp.rtconf.vrfs import SUPPORTED_VRF_RF
from os_ken.services.protocols.bgp.info_base.base import Filter
from os_ken.services.protocols.bgp.info_base.ipv4 import Ipv4Path
from os_ken.services.protocols.bgp.info_base.ipv6 import Ipv6Path
from os_ken.services.protocols.bgp.info_base.vpnv4 import Vpnv4Path
from os_ken.services.protocols.bgp.info_base.vpnv6 import Vpnv6Path
from os_ken.services.protocols.bgp.info_base.evpn import EvpnPath
def evpn_prefix_add(self, route_type, route_dist, esi=0, ethernet_tag_id=None, mac_addr=None, ip_addr=None, ip_prefix=None, gw_ip_addr=None, vni=None, next_hop=None, tunnel_type=None, pmsi_tunnel_type=None, redundancy_mode=None, tunnel_endpoint_ip=None, mac_mobility=None):
    """ This method adds a new EVPN route to be advertised.

        ``route_type`` specifies one of the EVPN route type name.
        This parameter must be one of the following.

        - EVPN_ETH_AUTO_DISCOVERY   = 'eth_ad'
        - EVPN_MAC_IP_ADV_ROUTE     = 'mac_ip_adv'
        - EVPN_MULTICAST_ETAG_ROUTE = 'multicast_etag'
        - EVPN_ETH_SEGMENT          = 'eth_seg'
        - EVPN_IP_PREFIX_ROUTE      = 'ip_prefix'

        ``route_dist`` specifies a route distinguisher value.

        ``esi`` is an value to specify the Ethernet Segment Identifier.
        0 is the default and denotes a single-homed site.
        If you want to advertise esi other than 0,
        it must be set as dictionary type.
        If esi is dictionary type, 'type' key must be set
        and specifies ESI type.
        For the supported ESI type, see :py:mod:`os_ken.lib.packet.bgp.EvpnEsi`.
        The remaining arguments are the same as that for
        the corresponding class.

        ``ethernet_tag_id`` specifies the Ethernet Tag ID.

        ``mac_addr`` specifies a MAC address to advertise.

        ``ip_addr`` specifies an IPv4 or IPv6 address to advertise.

        ``ip_prefix`` specifies an IPv4 or IPv6 prefix to advertise.

        ``gw_ip_addr`` specifies an IPv4 or IPv6 address of
        gateway to advertise.

        ``vni`` specifies an Virtual Network Identifier for VXLAN
        or Virtual Subnet Identifier for NVGRE.
        If tunnel_type is not TUNNEL_TYPE_VXLAN or TUNNEL_TYPE_NVGRE,
        this field is ignored.

        ``next_hop`` specifies the next hop address for this prefix.

        ``tunnel_type`` specifies the data plane encapsulation type
        to advertise. By the default, this attribute is not advertised.
        The supported encapsulation types are following.

        - TUNNEL_TYPE_VXLAN = 'vxlan'
        - TUNNEL_TYPE_NVGRE = 'nvgre

        ``pmsi_tunnel_type`` specifies the type of the PMSI tunnel attribute
        used to encode the multicast tunnel identifier.
        This attribute is advertised only if route_type is
        EVPN_MULTICAST_ETAG_ROUTE and not advertised by the default.
        This attribute can also carry vni if tunnel_type is specified.
        The supported PMSI tunnel types are following.

        - PMSI_TYPE_NO_TUNNEL_INFO = 0
        - PMSI_TYPE_INGRESS_REP    = 6

        ``redundancy_mode`` specifies a redundancy mode type.
        This attribute is advertised only if route_type is
        EVPN_ETH_AUTO_DISCOVERY and not advertised by the default.
        The supported redundancy mode types are following.

        - REDUNDANCY_MODE_ALL_ACTIVE    = 'all_active'
        - REDUNDANCY_MODE_SINGLE_ACTIVE = 'single_active'

        ``tunnel_endpoint_ip`` specifies a VTEP IP address other than the
        local router ID. This attribute is advertised only if route_type is
        EVPN_MULTICAST_ETAG_ROUTE, and defaults to the local router ID.

        ``mac_mobility`` specifies an optional integer sequence number to use
        in a MAC Mobility extended community field. The special value '-1' can
        be used to set the STATIC flag with a 0-value sequence number.

        """
    func_name = 'evpn_prefix.add_local'
    if not next_hop:
        next_hop = '0.0.0.0'
    kwargs = {EVPN_ROUTE_TYPE: route_type, ROUTE_DISTINGUISHER: route_dist, NEXT_HOP: next_hop}
    if tunnel_type in [TUNNEL_TYPE_VXLAN, TUNNEL_TYPE_NVGRE]:
        kwargs[TUNNEL_TYPE] = tunnel_type
    elif tunnel_type is not None:
        raise ValueError('Unsupported tunnel type: %s' % tunnel_type)
    if route_type == EVPN_ETH_AUTO_DISCOVERY:
        kwargs.update({EVPN_ESI: esi, EVPN_ETHERNET_TAG_ID: ethernet_tag_id})
        if vni is not None:
            kwargs[EVPN_VNI] = vni
        if redundancy_mode in [REDUNDANCY_MODE_ALL_ACTIVE, REDUNDANCY_MODE_SINGLE_ACTIVE]:
            kwargs[REDUNDANCY_MODE] = redundancy_mode
        elif redundancy_mode is not None:
            raise ValueError('Unsupported Redundancy Mode: %s' % redundancy_mode)
    elif route_type == EVPN_MAC_IP_ADV_ROUTE:
        kwargs.update({EVPN_ESI: esi, EVPN_ETHERNET_TAG_ID: ethernet_tag_id, MAC_ADDR: mac_addr, IP_ADDR: ip_addr})
        if tunnel_type in [TUNNEL_TYPE_VXLAN, TUNNEL_TYPE_NVGRE]:
            kwargs[EVPN_VNI] = vni
        if mac_mobility is not None:
            kwargs[MAC_MOBILITY] = int(mac_mobility)
    elif route_type == EVPN_MULTICAST_ETAG_ROUTE:
        kwargs.update({EVPN_ETHERNET_TAG_ID: ethernet_tag_id, IP_ADDR: ip_addr})
        if tunnel_type in [TUNNEL_TYPE_VXLAN, TUNNEL_TYPE_NVGRE]:
            kwargs[EVPN_VNI] = vni
        if pmsi_tunnel_type in [PMSI_TYPE_NO_TUNNEL_INFO, PMSI_TYPE_INGRESS_REP]:
            kwargs[PMSI_TUNNEL_TYPE] = pmsi_tunnel_type
        elif pmsi_tunnel_type is not None:
            raise ValueError('Unsupported PMSI tunnel type: %s' % pmsi_tunnel_type)
        if tunnel_endpoint_ip is not None:
            kwargs[TUNNEL_ENDPOINT_IP] = tunnel_endpoint_ip
    elif route_type == EVPN_ETH_SEGMENT:
        kwargs.update({EVPN_ESI: esi, IP_ADDR: ip_addr})
    elif route_type == EVPN_IP_PREFIX_ROUTE:
        kwargs.update({EVPN_ESI: esi, EVPN_ETHERNET_TAG_ID: ethernet_tag_id, IP_PREFIX: ip_prefix, GW_IP_ADDR: gw_ip_addr})
        if tunnel_type in [TUNNEL_TYPE_VXLAN, TUNNEL_TYPE_NVGRE]:
            kwargs[EVPN_VNI] = vni
        if mac_mobility is not None:
            kwargs[MAC_MOBILITY] = int(mac_mobility)
    else:
        raise ValueError('Unsupported EVPN route type: %s' % route_type)
    call(func_name, **kwargs)