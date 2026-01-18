import abc
from abc import ABCMeta
from abc import abstractmethod
from copy import copy
import logging
import functools
import netaddr
from os_ken.lib.packet.bgp import RF_IPv4_UC
from os_ken.lib.packet.bgp import RouteTargetMembershipNLRI
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_EXTENDED_COMMUNITIES
from os_ken.lib.packet.bgp import BGPPathAttributeLocalPref
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_AS_PATH
from os_ken.services.protocols.bgp.base import OrderedDict
from os_ken.services.protocols.bgp.constants import VPN_TABLE
from os_ken.services.protocols.bgp.constants import VRF_TABLE
from os_ken.services.protocols.bgp.model import OutgoingRoute
from os_ken.services.protocols.bgp.processor import BPR_ONLY_PATH
from os_ken.services.protocols.bgp.processor import BPR_UNKNOWN
def cleanup_paths_for_peer(self, peer):
    """Remove old paths from whose source is `peer`

        Old paths have source version number that is less than current peer
        version number. Also removes sent paths to this peer.
        """
    LOG.debug('Cleaning paths from table %s for peer %s', self, peer)
    for dest in self.values():
        paths_deleted = dest.remove_old_paths_from_source(peer)
        had_sent = dest.remove_sent_route(peer)
        if had_sent:
            LOG.debug('Removed sent route %s for %s', dest.nlri, peer)
        if paths_deleted:
            self._signal_bus.dest_changed(dest)