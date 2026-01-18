import logging
import time
import random
from os_ken.base import app_manager
from os_ken.controller import event
from os_ken.controller import ofp_event
from os_ken.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER
from os_ken.controller.handler import set_ev_cls
from os_ken.exception import OSKenException
from os_ken.ofproto.ether import ETH_TYPE_IP, ETH_TYPE_ARP
from os_ken.ofproto import ofproto_v1_3
from os_ken.ofproto import inet
from os_ken.lib import hub
from os_ken.lib.packet import packet
from os_ken.lib.packet import ethernet
from os_ken.lib.packet import ipv4
from os_ken.lib.packet import udp
from os_ken.lib.packet import bfd
from os_ken.lib.packet import arp
from os_ken.lib.packet.arp import ARP_REQUEST, ARP_REPLY
@staticmethod
def bfd_parse(data):
    """
        Parse raw packet and return BFD class from packet library.
        """
    pkt = packet.Packet(data)
    i = iter(pkt)
    eth_pkt = next(i)
    assert isinstance(eth_pkt, ethernet.ethernet)
    ipv4_pkt = next(i)
    assert isinstance(ipv4_pkt, ipv4.ipv4)
    udp_pkt = next(i)
    assert isinstance(udp_pkt, udp.udp)
    udp_payload = next(i)
    return bfd.bfd.parser(udp_payload)[0]