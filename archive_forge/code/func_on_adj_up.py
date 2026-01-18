from os_ken.services.protocols.bgp.base import Activity
from os_ken.lib import hub
from os_ken.lib.packet import bmp
from os_ken.lib.packet import bgp
import socket
import logging
from calendar import timegm
from os_ken.services.protocols.bgp.signals.emit import BgpSignalBus
from os_ken.services.protocols.bgp.info_base.ipv4 import Ipv4Path
from os_ken.lib.packet.bgp import BGPUpdate
from os_ken.lib.packet.bgp import BGPPathAttributeMpUnreachNLRI
def on_adj_up(self, data):
    peer = data['peer']
    msg = self._construct_peer_up_notification(peer)
    self._send(msg)