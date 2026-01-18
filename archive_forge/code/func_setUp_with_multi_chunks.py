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
def setUp_with_multi_chunks(self):
    self.s_flags = 0
    self.s_length = 16
    self.s_tsn_ack = 123456
    self.s_a_rwnd = 9876
    self.s_gapack_num = 0
    self.s_duptsn_num = 0
    self.s_gapacks = None
    self.s_duptsns = None
    self.sack = sctp.chunk_sack(tsn_ack=self.s_tsn_ack, a_rwnd=self.s_a_rwnd)
    self.d1_unordered = 0
    self.d1_begin = 1
    self.d1_end = 0
    self.d1_length = 16 + 10
    self.d1_tsn = 12345
    self.d1_sid = 1
    self.d1_seq = 0
    self.d1_payload_id = 0
    self.d1_payload_data = b'\x01\x02\x03\x04\x05\x06\x07\x08\t\n'
    self.data1 = sctp.chunk_data(begin=self.d1_begin, tsn=self.d1_tsn, sid=self.d1_sid, payload_data=self.d1_payload_data)
    self.d2_unordered = 0
    self.d2_begin = 0
    self.d2_end = 1
    self.d2_length = 16 + 10
    self.d2_tsn = 12346
    self.d2_sid = 1
    self.d2_seq = 1
    self.d2_payload_id = 0
    self.d2_payload_data = b'\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a'
    self.data2 = sctp.chunk_data(end=self.d2_end, tsn=self.d2_tsn, sid=self.d2_sid, seq=self.d2_seq, payload_data=self.d2_payload_data)
    self.chunks = [self.sack, self.data1, self.data2]
    self.sc = sctp.sctp(self.src_port, self.dst_port, self.vtag, self.csum, self.chunks)
    self.buf += b'\x03\x00\x00\x10\x00\x01\xe2@' + b'\x00\x00&\x94\x00\x00\x00\x00' + b'\x00\x02\x00\x1a\x00\x0009\x00\x01\x00\x00' + b'\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n' + b'\x00\x01\x00\x1a\x00\x000:\x00\x01\x00\x01' + b'\x00\x00\x00\x00\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a'