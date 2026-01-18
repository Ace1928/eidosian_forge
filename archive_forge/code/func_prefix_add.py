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
def prefix_add(self, prefix, next_hop=None, route_dist=None):
    """ This method adds a new prefix to be advertised.

        ``prefix`` must be the string representation of an IP network
        (e.g., 10.1.1.0/24).

        ``next_hop`` specifies the next hop address for this
        prefix. This parameter is necessary for only VPNv4 and VPNv6
        address families.

        ``route_dist`` specifies a route distinguisher value. This
        parameter is necessary for only VPNv4 and VPNv6 address
        families.
        """
    func_name = 'network.add'
    networks = {PREFIX: prefix}
    if next_hop:
        networks[NEXT_HOP] = next_hop
    if route_dist:
        func_name = 'prefix.add_local'
        networks[ROUTE_DISTINGUISHER] = route_dist
        rf, p = self._check_rf_and_normalize(prefix)
        networks[ROUTE_FAMILY] = rf
        networks[PREFIX] = p
        if rf == vrfs.VRF_RF_IPV6 and ip.valid_ipv4(next_hop):
            networks[NEXT_HOP] = str(netaddr.IPAddress(next_hop).ipv6())
    return call(func_name, **networks)