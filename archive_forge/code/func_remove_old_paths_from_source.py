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
def remove_old_paths_from_source(self, source):
    """Removes known old paths from *source*.

        Returns *True* if any of the known paths were found to be old and
        removed/deleted.
        """
    assert source and hasattr(source, 'version_num')
    removed_paths = []
    source_ver_num = source.version_num
    for path_idx in range(len(self._known_path_list) - 1, -1, -1):
        path = self._known_path_list[path_idx]
        if path.source == source and path.source_version_num < source_ver_num:
            del self._known_path_list[path_idx]
            removed_paths.append(path)
    return removed_paths