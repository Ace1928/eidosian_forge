import logging
from os_ken.lib.packet.bgp import RF_RTC_UC
from os_ken.lib.packet.bgp import RouteTargetMembershipNLRI
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_AS_PATH
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_ORIGIN
from os_ken.lib.packet.bgp import BGPPathAttributeAsPath
from os_ken.lib.packet.bgp import BGPPathAttributeOrigin
from os_ken.services.protocols.bgp.base import OrderedDict
from os_ken.services.protocols.bgp.info_base.rtc import RtcPath
def on_rt_filter_chg_sync_peer(self, peer, new_rts, old_rts, table):
    LOG.debug('RT Filter changed for peer %s, new_rts %s, old_rts %s ', peer, new_rts, old_rts)
    for dest in table.values():
        if not dest.best_path:
            continue
        desired_rts = set(dest.best_path.get_rts())
        if dest.was_sent_to(peer):
            if not desired_rts - old_rts:
                dest.withdraw_if_sent_to(peer)
        else:
            desired_rts.add(RouteTargetMembershipNLRI.DEFAULT_RT)
            if desired_rts.intersection(new_rts):
                peer.communicate_path(dest.best_path)