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
def comm_all_best_paths(self, peer):
    """Shares/communicates current best paths with this peers.

        Can be used to send initial updates after we have established session
        with `peer`.
        """
    LOG.debug('Communicating current best path for all afi/safi except 1/132')
    for route_family, table in self._table_manager.iter:
        if route_family == RF_RTC_UC:
            continue
        if peer.is_mbgp_cap_valid(route_family):
            for dest in table.values():
                if dest.best_path:
                    peer.communicate_path(dest.best_path)