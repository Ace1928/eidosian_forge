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
def comm_new_best_to_bgp_peers(self, new_best_path):
    """Communicates/enqueues given best path to be sent to all qualifying
        bgp peers.

        If this path came from iBGP peers, it is not sent to other iBGP peers.
        If this path has community-attribute, and if settings for recognize-
        well-know attributes is set, we do as per [RFC1997], and queue outgoing
        route only to qualifying BGP peers.
        """
    comm_attr = new_best_path.get_pattr(BGP_ATTR_TYPE_COMMUNITIES)
    if comm_attr:
        comm_attr_na = comm_attr.has_comm_attr(BGPPathAttributeCommunities.NO_ADVERTISE)
        if comm_attr_na:
            LOG.debug('New best path has community attr. NO_ADVERTISE = %s. Hence not advertising to any peer', comm_attr_na)
            return
    qualified_peers = self._collect_peers_of_interest(new_best_path)
    for peer in qualified_peers:
        peer.communicate_path(new_best_path)