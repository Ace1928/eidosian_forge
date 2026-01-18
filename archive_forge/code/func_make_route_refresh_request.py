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
def make_route_refresh_request(self, peer_ip, *route_families):
    """Request route-refresh for peer with `peer_ip` for given
        `route_families`.

        Will make route-refresh request for a given `route_family` only if such
        capability is supported and if peer is in ESTABLISHED state. Else, such
        requests are ignored. Raises appropriate error in other cases. If
        `peer_ip` is equal to 'all' makes refresh request to all valid peers.
        """
    LOG.debug('Route refresh requested for peer %s and route families %s', peer_ip, route_families)
    if not SUPPORTED_GLOBAL_RF.intersection(route_families):
        raise ValueError('Given route family(s) % is not supported.' % route_families)
    peer_list = []
    if peer_ip == 'all':
        peer_list.extend(self.get_peers_in_established())
    else:
        given_peer = self._peers.get(peer_ip)
        if not given_peer:
            raise ValueError('Invalid/unrecognized peer %s' % peer_ip)
        if not given_peer.in_established:
            raise ValueError('Peer currently do not have established session.')
        peer_list.append(given_peer)
    for peer in peer_list:
        peer.request_route_refresh(*route_families)
    return True