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
@set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
def lldp_packet_in_handler(self, ev):
    if not self.link_discovery:
        return
    msg = ev.msg
    try:
        src_dpid, src_port_no = LLDPPacket.lldp_parse(msg.data)
    except LLDPPacket.LLDPUnknownFormat:
        return
    dst_dpid = msg.datapath.id
    if msg.datapath.ofproto.OFP_VERSION == ofproto_v1_0.OFP_VERSION:
        dst_port_no = msg.in_port
    elif msg.datapath.ofproto.OFP_VERSION >= ofproto_v1_2.OFP_VERSION:
        dst_port_no = msg.match['in_port']
    else:
        LOG.error('cannot accept LLDP. unsupported version. %x', msg.datapath.ofproto.OFP_VERSION)
    src = self._get_port(src_dpid, src_port_no)
    if not src or src.dpid == dst_dpid:
        return
    try:
        self.ports.lldp_received(src)
    except KeyError:
        pass
    dst = self._get_port(dst_dpid, dst_port_no)
    if not dst:
        return
    link = Link(src, dst)
    if link not in self.links:
        self.send_event_to_observers(event.EventLinkAdd(link))
        host_to_del = []
        for host in self.hosts.values():
            if not self._is_edge_port(host.port):
                host_to_del.append(host.mac)
        for host_mac in host_to_del:
            del self.hosts[host_mac]
    if not self.links.update_link(src, dst):
        self.ports.move_front(dst)
        self.lldp_event.set()
    if self.explicit_drop:
        self._drop_packet(msg)