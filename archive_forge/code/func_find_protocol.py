import unittest
import logging
import struct
from struct import *
from os_ken.ofproto import ether
from os_ken.lib.packet.ethernet import ethernet
from os_ken.lib.packet.packet import Packet
from os_ken.lib.packet.arp import arp
from os_ken.lib.packet.vlan import vlan
from os_ken.lib import addrconv
def find_protocol(self, pkt, name):
    for p in pkt.protocols:
        if p.protocol_name == name:
            return p