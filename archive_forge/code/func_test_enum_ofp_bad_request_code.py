import unittest
import logging
from os_ken.ofproto.ofproto_v1_2 import *
def test_enum_ofp_bad_request_code(self):
    self.assertEqual(OFPBRC_BAD_VERSION, 0)
    self.assertEqual(OFPBRC_BAD_TYPE, 1)
    self.assertEqual(OFPBRC_BAD_STAT, 2)
    self.assertEqual(OFPBRC_BAD_EXPERIMENTER, 3)
    self.assertEqual(OFPBRC_BAD_EXP_TYPE, 4)
    self.assertEqual(OFPBRC_EPERM, 5)
    self.assertEqual(OFPBRC_BAD_LEN, 6)
    self.assertEqual(OFPBRC_BUFFER_EMPTY, 7)
    self.assertEqual(OFPBRC_BUFFER_UNKNOWN, 8)
    self.assertEqual(OFPBRC_BAD_TABLE_ID, 9)
    self.assertEqual(OFPBRC_IS_SLAVE, 10)
    self.assertEqual(OFPBRC_BAD_PORT, 11)
    self.assertEqual(OFPBRC_BAD_PACKET, 12)