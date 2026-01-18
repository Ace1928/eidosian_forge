import logging
import netaddr
from os_ken.services.protocols.bgp.base import SUPPORTED_GLOBAL_RF
from os_ken.services.protocols.bgp.model import OutgoingRoute
from os_ken.services.protocols.bgp.peer import Peer
from os_ken.lib.packet.bgp import BGPPathAttributeCommunities
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_MULTI_EXIT_DISC
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_COMMUNITIES
from os_ken.lib.packet.bgp import RF_RTC_UC
from os_ken.lib.packet.bgp import RouteTargetMembershipNLRI
from os_ken.services.protocols.bgp.utils.bgp \
def resend_sent(self, route_family, peer):
    """For given `peer` re-send sent paths.

        Parameters:
            - `route-family`: (RouteFamily) of the sent paths to re-send
            - `peer`: (Peer) peer for which we need to re-send sent paths
        """
    if peer not in self._peers.values():
        raise ValueError('Could not find given peer (%s)' % peer)
    if route_family not in SUPPORTED_GLOBAL_RF:
        raise ValueError('Given route family (%s) is not supported.' % route_family)
    table = self._table_manager.get_global_table_by_route_family(route_family)
    for destination in table.values():
        sent_routes = destination.sent_routes
        if sent_routes is None or len(sent_routes) == 0:
            continue
        for sent_route in sent_routes:
            if sent_route.sent_peer == peer:
                p = sent_route.path
                if p.med_set_by_target_neighbor or p.get_pattr(BGP_ATTR_TYPE_MULTI_EXIT_DISC) is None:
                    sent_route.path = clone_path_and_update_med_for_target_neighbor(sent_route.path, peer.med)
                ogr = OutgoingRoute(sent_route.path, for_route_refresh=True)
                peer.enque_outgoing_msg(ogr)