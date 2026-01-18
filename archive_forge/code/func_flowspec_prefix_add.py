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
def flowspec_prefix_add(self, flowspec_family, rules, route_dist=None, actions=None):
    """ This method adds a new Flow Specification prefix to be advertised.

        ``flowspec_family`` specifies one of the flowspec family name.
        This parameter must be one of the following.

        - FLOWSPEC_FAMILY_IPV4  = 'ipv4fs'
        - FLOWSPEC_FAMILY_IPV6  = 'ipv6fs'
        - FLOWSPEC_FAMILY_VPNV4 = 'vpnv4fs'
        - FLOWSPEC_FAMILY_VPNV6 = 'vpnv6fs'
        - FLOWSPEC_FAMILY_L2VPN = 'l2vpnfs'

        ``rules`` specifies NLRIs of Flow Specification as
        a dictionary type value.
        For the supported NLRI types and arguments,
        see `from_user()` method of the following classes.

        - :py:mod:`os_ken.lib.packet.bgp.FlowSpecIPv4NLRI`
        - :py:mod:`os_ken.lib.packet.bgp.FlowSpecIPv6NLRI`
        - :py:mod:`os_ken.lib.packet.bgp.FlowSpecVPNv4NLRI`
        - :py:mod:`os_ken.lib.packet.bgp.FlowSpecVPNv6NLRI`
        - :py:mod:`os_ken.lib.packet.bgp.FlowSpecL2VPNNLRI`

        ``route_dist`` specifies a route distinguisher value.
        This parameter is required only if flowspec_family is one of the
        following address family.

        - FLOWSPEC_FAMILY_VPNV4 = 'vpnv4fs'
        - FLOWSPEC_FAMILY_VPNV6 = 'vpnv6fs'
        - FLOWSPEC_FAMILY_L2VPN = 'l2vpnfs'

        ``actions`` specifies Traffic Filtering Actions of
        Flow Specification as a dictionary type value.
        The keys are "ACTION_NAME" for each action class and
        values are used for the arguments to that class.
        For the supported "ACTION_NAME" and arguments,
        see the following table.

        =============== ===============================================================
        ACTION_NAME     Action Class
        =============== ===============================================================
        traffic_rate    :py:mod:`os_ken.lib.packet.bgp.BGPFlowSpecTrafficRateCommunity`
        traffic_action  :py:mod:`os_ken.lib.packet.bgp.BGPFlowSpecTrafficActionCommunity`
        redirect        :py:mod:`os_ken.lib.packet.bgp.BGPFlowSpecRedirectCommunity`
        traffic_marking :py:mod:`os_ken.lib.packet.bgp.BGPFlowSpecTrafficMarkingCommunity`
        vlan_action     :py:mod:`os_ken.lib.packet.bgp.BGPFlowSpecVlanActionCommunity`
        tpid_action     :py:mod:`os_ken.lib.packet.bgp.BGPFlowSpecTPIDActionCommunity`
        =============== ===============================================================

        Example(IPv4)::

            >>> speaker = BGPSpeaker(as_number=65001, router_id='172.17.0.1')
            >>> speaker.neighbor_add(address='172.17.0.2',
            ...                      remote_as=65002,
            ...                      enable_ipv4fs=True)
            >>> speaker.flowspec_prefix_add(
            ...     flowspec_family=FLOWSPEC_FAMILY_IPV4,
            ...     rules={
            ...         'dst_prefix': '10.60.1.0/24'
            ...     },
            ...     actions={
            ...         'traffic_marking': {
            ...             'dscp': 24
            ...         }
            ...     }
            ... )

        Example(VPNv4)::

            >>> speaker = BGPSpeaker(as_number=65001, router_id='172.17.0.1')
            >>> speaker.neighbor_add(address='172.17.0.2',
            ...                      remote_as=65002,
            ...                      enable_vpnv4fs=True)
            >>> speaker.vrf_add(route_dist='65001:100',
            ...                 import_rts=['65001:100'],
            ...                 export_rts=['65001:100'],
            ...                 route_family=RF_VPNV4_FLOWSPEC)
            >>> speaker.flowspec_prefix_add(
            ...     flowspec_family=FLOWSPEC_FAMILY_VPNV4,
            ...     route_dist='65000:100',
            ...     rules={
            ...         'dst_prefix': '10.60.1.0/24'
            ...     },
            ...     actions={
            ...         'traffic_marking': {
            ...             'dscp': 24
            ...         }
            ...     }
            ... )
        """
    func_name = 'flowspec.add'
    kwargs = {FLOWSPEC_FAMILY: flowspec_family, FLOWSPEC_RULES: rules, FLOWSPEC_ACTIONS: actions or {}}
    if flowspec_family in [FLOWSPEC_FAMILY_VPNV4, FLOWSPEC_FAMILY_VPNV6, FLOWSPEC_FAMILY_L2VPN]:
        func_name = 'flowspec.add_local'
        kwargs.update({ROUTE_DISTINGUISHER: route_dist})
    call(func_name, **kwargs)