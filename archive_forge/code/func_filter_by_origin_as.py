import logging
from os_ken.lib.packet.bgp import RF_RTC_UC
from os_ken.lib.packet.bgp import RouteTargetMembershipNLRI
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_AS_PATH
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_ORIGIN
from os_ken.lib.packet.bgp import BGPPathAttributeAsPath
from os_ken.lib.packet.bgp import BGPPathAttributeOrigin
from os_ken.services.protocols.bgp.base import OrderedDict
from os_ken.services.protocols.bgp.info_base.rtc import RtcPath
def filter_by_origin_as(self, new_best_path, qualified_peers):
    path_rf = new_best_path.route_family
    if path_rf != RF_RTC_UC or new_best_path.source is not None:
        return qualified_peers
    else:
        filtered_qualified_peers = []
        rt_origin_as = new_best_path.nlri.origin_as
        for qualified_peer in qualified_peers:
            neigh_conf = self._neighbors_conf.get_neighbor_conf(qualified_peer.ip_address)
            if neigh_conf.rtc_as == rt_origin_as:
                filtered_qualified_peers.append(qualified_peer)
        return filtered_qualified_peers