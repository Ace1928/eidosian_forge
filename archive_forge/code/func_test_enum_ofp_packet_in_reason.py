import unittest
import logging
from os_ken.ofproto.ofproto_v1_2 import *
def test_enum_ofp_packet_in_reason(self):
    self.assertEqual(OFPR_NO_MATCH, 0)
    self.assertEqual(OFPR_ACTION, 1)
    self.assertEqual(OFPR_INVALID_TTL, 2)