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
def host_discovery_packet_in_handler(self, ev):
    msg = ev.msg
    eth, pkt_type, pkt_data = ethernet.ethernet.parser(msg.data)
    if eth.ethertype in (ETH_TYPE_LLDP, ETH_TYPE_CFM):
        return
    datapath = msg.datapath
    dpid = datapath.id
    port_no = -1
    if msg.datapath.ofproto.OFP_VERSION == ofproto_v1_0.OFP_VERSION:
        port_no = msg.in_port
    else:
        port_no = msg.match['in_port']
    port = self._get_port(dpid, port_no)
    if not port:
        return
    if not self._is_edge_port(port):
        return
    host_mac = eth.src
    host = Host(host_mac, port)
    if host_mac not in self.hosts:
        self.hosts.add(host)
        ev = event.EventHostAdd(host)
        self.send_event_to_observers(ev)
    elif self.hosts[host_mac].port != port:
        ev = event.EventHostMove(src=self.hosts[host_mac], dst=host)
        self.hosts[host_mac] = host
        self.send_event_to_observers(ev)
    if eth.ethertype == ether_types.ETH_TYPE_ARP:
        arp_pkt, _, _ = pkt_type.parser(pkt_data)
        self.hosts.update_ip(host, ip_v4=arp_pkt.src_ip)
    elif eth.ethertype == ether_types.ETH_TYPE_IP:
        ipv4_pkt, _, _ = pkt_type.parser(pkt_data)
        self.hosts.update_ip(host, ip_v4=ipv4_pkt.src)
    elif eth.ethertype == ether_types.ETH_TYPE_IPV6:
        ipv6_pkt, _, _ = pkt_type.parser(pkt_data)
        self.hosts.update_ip(host, ip_v6=ipv6_pkt.src)