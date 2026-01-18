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
def setUp_with_sack(self):
    self.flags = 0
    self.length = 16 + 2 * 2 * 5 + 4 * 5
    self.tsn_ack = 123456
    self.a_rwnd = 9876
    self.gapack_num = 5
    self.duptsn_num = 5
    self.gapacks = [[2, 3], [10, 12], [20, 24], [51, 52], [62, 63]]
    self.duptsns = [123458, 123466, 123476, 123507, 123518]
    self.sack = sctp.chunk_sack(tsn_ack=self.tsn_ack, a_rwnd=self.a_rwnd, gapack_num=self.gapack_num, duptsn_num=self.duptsn_num, gapacks=self.gapacks, duptsns=self.duptsns)
    self.chunks = [self.sack]
    self.sc = sctp.sctp(self.src_port, self.dst_port, self.vtag, self.csum, self.chunks)
    self.buf += b'\x03\x00\x008\x00\x01\xe2@' + b'\x00\x00&\x94\x00\x05\x00\x05' + b'\x00\x02\x00\x03\x00\n\x00\x0c\x00\x14\x00\x18' + b'\x003\x004\x00>\x00?' + b'\x00\x01\xe2B\x00\x01\xe2J\x00\x01\xe2T' + b'\x00\x01\xe2s\x00\x01\xe2~'