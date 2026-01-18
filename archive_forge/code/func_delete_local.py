import logging
from os_ken.lib.packet.bgp import EvpnEsi
from os_ken.lib.packet.bgp import EvpnNLRI
from os_ken.lib.packet.bgp import EvpnEthernetAutoDiscoveryNLRI
from os_ken.lib.packet.bgp import EvpnMacIPAdvertisementNLRI
from os_ken.lib.packet.bgp import EvpnInclusiveMulticastEthernetTagNLRI
from os_ken.lib.packet.bgp import EvpnEthernetSegmentNLRI
from os_ken.lib.packet.bgp import EvpnIpPrefixNLRI
from os_ken.lib.packet.bgp import BGPPathAttributePmsiTunnel
from os_ken.lib.packet.bgp import FlowSpecIPv4NLRI
from os_ken.lib.packet.bgp import FlowSpecIPv6NLRI
from os_ken.lib.packet.bgp import FlowSpecVPNv4NLRI
from os_ken.lib.packet.bgp import FlowSpecVPNv6NLRI
from os_ken.lib.packet.bgp import FlowSpecL2VPNNLRI
from os_ken.lib.packet.bgp import BGPFlowSpecTrafficRateCommunity
from os_ken.lib.packet.bgp import BGPFlowSpecTrafficActionCommunity
from os_ken.lib.packet.bgp import BGPFlowSpecRedirectCommunity
from os_ken.lib.packet.bgp import BGPFlowSpecTrafficMarkingCommunity
from os_ken.lib.packet.bgp import BGPFlowSpecVlanActionCommunity
from os_ken.lib.packet.bgp import BGPFlowSpecTPIDActionCommunity
from os_ken.services.protocols.bgp.api.base import EVPN_ROUTE_TYPE
from os_ken.services.protocols.bgp.api.base import EVPN_ESI
from os_ken.services.protocols.bgp.api.base import EVPN_ETHERNET_TAG_ID
from os_ken.services.protocols.bgp.api.base import REDUNDANCY_MODE
from os_ken.services.protocols.bgp.api.base import MAC_ADDR
from os_ken.services.protocols.bgp.api.base import IP_ADDR
from os_ken.services.protocols.bgp.api.base import IP_PREFIX
from os_ken.services.protocols.bgp.api.base import GW_IP_ADDR
from os_ken.services.protocols.bgp.api.base import MPLS_LABELS
from os_ken.services.protocols.bgp.api.base import NEXT_HOP
from os_ken.services.protocols.bgp.api.base import PREFIX
from os_ken.services.protocols.bgp.api.base import RegisterWithArgChecks
from os_ken.services.protocols.bgp.api.base import ROUTE_DISTINGUISHER
from os_ken.services.protocols.bgp.api.base import VPN_LABEL
from os_ken.services.protocols.bgp.api.base import EVPN_VNI
from os_ken.services.protocols.bgp.api.base import TUNNEL_TYPE
from os_ken.services.protocols.bgp.api.base import PMSI_TUNNEL_TYPE
from os_ken.services.protocols.bgp.api.base import MAC_MOBILITY
from os_ken.services.protocols.bgp.api.base import TUNNEL_ENDPOINT_IP
from os_ken.services.protocols.bgp.api.base import FLOWSPEC_FAMILY
from os_ken.services.protocols.bgp.api.base import FLOWSPEC_RULES
from os_ken.services.protocols.bgp.api.base import FLOWSPEC_ACTIONS
from os_ken.services.protocols.bgp.base import add_bgp_error_metadata
from os_ken.services.protocols.bgp.base import PREFIX_ERROR_CODE
from os_ken.services.protocols.bgp.base import validate
from os_ken.services.protocols.bgp.core import BgpCoreError
from os_ken.services.protocols.bgp.core_manager import CORE_MANAGER
from os_ken.services.protocols.bgp.rtconf.base import ConfigValueError
from os_ken.services.protocols.bgp.rtconf.base import RuntimeConfigError
from os_ken.services.protocols.bgp.rtconf.vrfs import VRF_RF
from os_ken.services.protocols.bgp.rtconf.vrfs import VRF_RF_IPV4
from os_ken.services.protocols.bgp.rtconf.vrfs import VRF_RF_L2_EVPN
from os_ken.services.protocols.bgp.utils import validation
@RegisterWithArgChecks(name='prefix.delete_local', req_args=[ROUTE_DISTINGUISHER, PREFIX], opt_args=[VRF_RF])
def delete_local(route_dist, prefix, route_family=VRF_RF_IPV4):
    """Deletes/withdraws *prefix* from VRF identified by *route_dist* and
    source as network controller.
    """
    try:
        tm = CORE_MANAGER.get_core_service().table_manager
        tm.update_vrf_table(route_dist, prefix, route_family=route_family, is_withdraw=True)
        return [{ROUTE_DISTINGUISHER: route_dist, PREFIX: prefix, VRF_RF: route_family}]
    except BgpCoreError as e:
        raise PrefixError(desc=e)