import inspect
import logging
import struct
import unittest
from os_ken.lib import addrconv
from os_ken.lib.packet import packet
from os_ken.lib.packet import ethernet
from os_ken.lib.packet import ipv4
from os_ken.lib.packet import sctp
from os_ken.ofproto import ether
from os_ken.ofproto import inet
def setUp_with_heartbeat_ack(self):
    self.flags = 0
    self.length = 4 + 12
    self.p_heartbeat = sctp.param_heartbeat(b'\xff\xee\xdd\xcc\xbb\xaa\x99\x88')
    self.heartbeat_ack = sctp.chunk_heartbeat_ack(info=self.p_heartbeat)
    self.chunks = [self.heartbeat_ack]
    self.sc = sctp.sctp(self.src_port, self.dst_port, self.vtag, self.csum, self.chunks)
    self.buf += b'\x05\x00\x00\x10' + b'\x00\x01\x00\x0c' + b'\xff\xee\xdd\xcc\xbb\xaa\x99\x88'