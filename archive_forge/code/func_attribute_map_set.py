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
def attribute_map_set(self, address, attribute_maps, route_dist=None, route_family=RF_VPN_V4):
    """This method sets attribute mapping to a neighbor.
        attribute mapping can be used when you want to apply
        attribute to BGPUpdate under specific conditions.

        ``address`` specifies the IP address of the neighbor

        ``attribute_maps`` specifies attribute_map list that are used
        before paths are advertised. All the items in the list must
        be an instance of AttributeMap class

        ``route_dist`` specifies route dist in which attribute_maps
        are added.

        ``route_family`` specifies route family of the VRF.
        This parameter must be one of the following.

        - RF_VPN_V4 (default) = 'ipv4'
        - RF_VPN_V6           = 'ipv6'

        We can set AttributeMap to a neighbor as follows::

            pref_filter = PrefixFilter('192.168.103.0/30',
                                       PrefixFilter.POLICY_PERMIT)

            attribute_map = AttributeMap([pref_filter],
                                         AttributeMap.ATTR_LOCAL_PREF, 250)

            speaker.attribute_map_set('192.168.50.102', [attribute_map])
        """
    if route_family not in SUPPORTED_VRF_RF:
        raise ValueError('Unsupported route_family: %s' % route_family)
    func_name = 'neighbor.attribute_map.set'
    param = {neighbors.IP_ADDRESS: address, neighbors.ATTRIBUTE_MAP: attribute_maps}
    if route_dist is not None:
        param[vrfs.ROUTE_DISTINGUISHER] = route_dist
        param[vrfs.VRF_RF] = route_family
    call(func_name, **param)