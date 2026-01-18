import binascii
import unittest
import struct
from os_ken import exception
from os_ken.ofproto import ofproto_common, ofproto_parser
from os_ken.ofproto import ofproto_v1_0, ofproto_v1_0_parser
import logging
def testHello(self):
    version, msg_type, msg_len, xid = ofproto_parser.header(self.bufHello)
    self.assertEqual(version, 1)
    self.assertEqual(msg_type, 0)
    self.assertEqual(msg_len, 8)
    self.assertEqual(xid, 1)