from collections import namedtuple
import itertools
import logging
import socket
import time
import traceback
from os_ken.services.protocols.bgp.base import Activity
from os_ken.services.protocols.bgp.base import Sink
from os_ken.services.protocols.bgp.base import Source
from os_ken.services.protocols.bgp.base import SUPPORTED_GLOBAL_RF
from os_ken.services.protocols.bgp import constants as const
from os_ken.services.protocols.bgp.model import OutgoingRoute
from os_ken.services.protocols.bgp.model import SentRoute
from os_ken.services.protocols.bgp.info_base.base import PrefixFilter
from os_ken.services.protocols.bgp.info_base.base import AttributeMap
from os_ken.services.protocols.bgp.model import ReceivedRoute
from os_ken.services.protocols.bgp.net_ctrl import NET_CONTROLLER
from os_ken.services.protocols.bgp.rtconf.neighbors import NeighborConfListener
from os_ken.services.protocols.bgp.rtconf.neighbors import CONNECT_MODE_PASSIVE
from os_ken.services.protocols.bgp.signals.emit import BgpSignalBus
from os_ken.services.protocols.bgp.speaker import BgpProtocol
from os_ken.services.protocols.bgp.info_base.ipv4 import Ipv4Path
from os_ken.services.protocols.bgp.info_base.vpnv4 import Vpnv4Path
from os_ken.services.protocols.bgp.info_base.vpnv6 import Vpnv6Path
from os_ken.services.protocols.bgp.rtconf.vrfs import VRF_RF_IPV4, VRF_RF_IPV6
from os_ken.services.protocols.bgp.utils import bgp as bgp_utils
from os_ken.services.protocols.bgp.utils.evtlet import EventletIOFactory
from os_ken.services.protocols.bgp.utils import stats
from os_ken.services.protocols.bgp.utils.validation import is_valid_old_asn
from os_ken.lib.packet import bgp
from os_ken.lib.packet.bgp import RouteFamily
from os_ken.lib.packet.bgp import RF_IPv4_UC
from os_ken.lib.packet.bgp import RF_IPv6_UC
from os_ken.lib.packet.bgp import RF_IPv4_VPN
from os_ken.lib.packet.bgp import RF_IPv6_VPN
from os_ken.lib.packet.bgp import RF_IPv4_FLOWSPEC
from os_ken.lib.packet.bgp import RF_VPNv4_FLOWSPEC
from os_ken.lib.packet.bgp import RF_RTC_UC
from os_ken.lib.packet.bgp import get_rf
from os_ken.lib.packet.bgp import BGPOpen
from os_ken.lib.packet.bgp import BGPUpdate
from os_ken.lib.packet.bgp import BGPRouteRefresh
from os_ken.lib.packet.bgp import BGP_ERROR_CEASE
from os_ken.lib.packet.bgp import BGP_ERROR_SUB_ADMINISTRATIVE_SHUTDOWN
from os_ken.lib.packet.bgp import BGP_ERROR_SUB_CONNECTION_COLLISION_RESOLUTION
from os_ken.lib.packet.bgp import BGP_MSG_UPDATE
from os_ken.lib.packet.bgp import BGP_MSG_KEEPALIVE
from os_ken.lib.packet.bgp import BGP_MSG_ROUTE_REFRESH
from os_ken.lib.packet.bgp import BGPPathAttributeNextHop
from os_ken.lib.packet.bgp import BGPPathAttributeAsPath
from os_ken.lib.packet.bgp import BGPPathAttributeAs4Path
from os_ken.lib.packet.bgp import BGPPathAttributeLocalPref
from os_ken.lib.packet.bgp import BGPPathAttributeExtendedCommunities
from os_ken.lib.packet.bgp import BGPPathAttributeOriginatorId
from os_ken.lib.packet.bgp import BGPPathAttributeClusterList
from os_ken.lib.packet.bgp import BGPPathAttributeMpReachNLRI
from os_ken.lib.packet.bgp import BGPPathAttributeMpUnreachNLRI
from os_ken.lib.packet.bgp import BGPPathAttributeCommunities
from os_ken.lib.packet.bgp import BGPPathAttributeMultiExitDisc
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_ORIGIN
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_AGGREGATOR
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_AS4_AGGREGATOR
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_AS_PATH
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_AS4_PATH
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_NEXT_HOP
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_MP_REACH_NLRI
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_MP_UNREACH_NLRI
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_MULTI_EXIT_DISC
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_COMMUNITIES
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_ORIGINATOR_ID
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_CLUSTER_LIST
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_EXTENDED_COMMUNITIES
from os_ken.lib.packet.bgp import BGP_ATTR_TYEP_PMSI_TUNNEL_ATTRIBUTE
from os_ken.lib.packet.bgp import BGPTwoOctetAsSpecificExtendedCommunity
from os_ken.lib.packet.bgp import BGPIPv4AddressSpecificExtendedCommunity
from os_ken.lib.packet import safi as subaddr_family
def communicate_path(self, path):
    """Communicates `path` to this peer if it qualifies.

        Checks if `path` should be shared/communicated with this peer according
        to various conditions: like bgp state, transmit side loop, local and
        remote AS path, community attribute, etc.
        """
    LOG.debug('Peer %s asked to communicate path', self)
    if not path:
        raise ValueError('Invalid path %s given.' % path)
    if not self.in_established():
        LOG.debug('Skipping sending path as peer is not in ESTABLISHED state %s', path)
        return
    path_rf = path.route_family
    if not (self.is_mpbgp_cap_valid(path_rf) or path_rf in [RF_IPv4_UC, RF_IPv6_UC]):
        LOG.debug('Skipping sending path as %s route family is not available for this session', path_rf)
        return
    if path_rf != RF_RTC_UC and self.is_mpbgp_cap_valid(RF_RTC_UC):
        rtfilter = self._peer_manager.curr_peer_rtfilter(self)
        if rtfilter and (not path.has_rts_in(rtfilter)):
            LOG.debug('Skipping sending path as rffilter %s and path rts %s have no RT in common', rtfilter, path.get_rts())
            return
    as_path = path.get_pattr(BGP_ATTR_TYPE_AS_PATH)
    if as_path and as_path.has_matching_leftmost(self.remote_as):
        LOG.debug('Skipping sending path as AS_PATH has peer AS %s', self.remote_as)
        return
    if self.is_route_server_client:
        outgoing_route = OutgoingRoute(path)
        self.enque_outgoing_msg(outgoing_route)
    if self._neigh_conf.multi_exit_disc:
        med_attr = path.get_pattr(BGP_ATTR_TYPE_MULTI_EXIT_DISC)
        if not med_attr:
            path = bgp_utils.clone_path_and_update_med_for_target_neighbor(path, self._neigh_conf.multi_exit_disc)
    if path.source is None:
        outgoing_route = OutgoingRoute(path)
        self.enque_outgoing_msg(outgoing_route)
    elif self != path.source or self.remote_as != path.source.remote_as:
        if self.remote_as == self._core_service.asn and self.remote_as == path.source.remote_as and isinstance(path.source, Peer) and (not path.source.is_route_reflector_client) and (not self.is_route_reflector_client):
            LOG.debug('Skipping sending iBGP route to iBGP peer %s AS %s', self.ip_address, self.remote_as)
            return
        comm_attr = path.get_pattr(BGP_ATTR_TYPE_COMMUNITIES)
        if comm_attr:
            comm_attr_na = comm_attr.has_comm_attr(BGPPathAttributeCommunities.NO_ADVERTISE)
            if comm_attr_na:
                LOG.debug('Path has community attr. NO_ADVERTISE = %s. Hence not advertising to peer', comm_attr_na)
                return
            comm_attr_ne = comm_attr.has_comm_attr(BGPPathAttributeCommunities.NO_EXPORT)
            comm_attr_nes = comm_attr.has_comm_attr(BGPPathAttributeCommunities.NO_EXPORT_SUBCONFED)
            if (comm_attr_nes or comm_attr_ne) and self.remote_as != self._core_service.asn:
                LOG.debug('Skipping sending UPDATE to peer: %s as per community attribute configuration', self)
                return
        outgoing_route = OutgoingRoute(path)
        self.enque_outgoing_msg(outgoing_route)
        LOG.debug('Enqueued outgoing route %s for peer %s', outgoing_route.path.nlri, self)