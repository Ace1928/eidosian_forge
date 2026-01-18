import binascii
import unittest
import struct
from os_ken import exception
from os_ken.ofproto import ofproto_common, ofproto_parser
from os_ken.ofproto import ofproto_v1_0, ofproto_v1_0_parser
import logging
def testPacketIn(self):
    version, msg_type, msg_len, xid = ofproto_parser.header(self.bufPacketIn)
    msg = ofproto_parser.msg(self, version, msg_type, msg_len, xid, self.bufPacketIn)
    LOG.debug(msg)
    self.assertTrue(isinstance(msg, ofproto_v1_0_parser.OFPPacketIn))