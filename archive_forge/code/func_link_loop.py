import logging
import struct
import time
from os_ken import cfg
from collections import defaultdict
from os_ken.topology import event
from os_ken.base import app_manager
from os_ken.controller import ofp_event
from os_ken.controller.handler import set_ev_cls
from os_ken.controller.handler import MAIN_DISPATCHER, DEAD_DISPATCHER
from os_ken.exception import OSKenException
from os_ken.lib import addrconv, hub
from os_ken.lib.mac import DONTCARE_STR
from os_ken.lib.dpid import dpid_to_str, str_to_dpid
from os_ken.lib.port_no import port_no_to_str
from os_ken.lib.packet import packet, ethernet
from os_ken.lib.packet import lldp, ether_types
from os_ken.ofproto.ether import ETH_TYPE_LLDP
from os_ken.ofproto.ether import ETH_TYPE_CFM
from os_ken.ofproto import nx_match
from os_ken.ofproto import ofproto_v1_0
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ofproto_v1_3
from os_ken.ofproto import ofproto_v1_4
def link_loop(self):
    while self.is_active:
        self.link_event.clear()
        now = time.time()
        deleted = []
        for link, timestamp in self.links.items():
            if timestamp + self.LINK_TIMEOUT < now:
                deleted.append(link)
        for link in deleted:
            self.links.link_down(link)
            self.send_event_to_observers(event.EventLinkDelete(link))
            dst = link.dst
            rev_link = Link(dst, link.src)
            if rev_link not in deleted:
                expire = now - self.LINK_TIMEOUT
                self.links.rev_link_set_timestamp(rev_link, expire)
                if dst in self.ports:
                    self.ports.move_front(dst)
                    self.lldp_event.set()
        self.link_event.wait(timeout=self.TIMEOUT_CHECK_PERIOD)