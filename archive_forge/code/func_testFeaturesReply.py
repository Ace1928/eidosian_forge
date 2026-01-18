import binascii
import unittest
import struct
from os_ken import exception
from os_ken.ofproto import ofproto_common, ofproto_parser
from os_ken.ofproto import ofproto_v1_0, ofproto_v1_0_parser
import logging
def testFeaturesReply(self):
    version, msg_type, msg_len, xid = ofproto_parser.header(self.bufFeaturesReply)
    msg = ofproto_parser.msg(self, version, msg_type, msg_len, xid, self.bufFeaturesReply)
    LOG.debug(msg)
    self.assertTrue(isinstance(msg, ofproto_v1_0_parser.OFPSwitchFeatures))
    LOG.debug(msg.ports[65534])
    self.assertTrue(isinstance(msg.ports[1], ofproto_v1_0_parser.OFPPhyPort))
    self.assertTrue(isinstance(msg.ports[2], ofproto_v1_0_parser.OFPPhyPort))
    self.assertTrue(isinstance(msg.ports[65534], ofproto_v1_0_parser.OFPPhyPort))