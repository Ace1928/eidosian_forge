import unittest
import logging
from os_ken.ofproto.ofproto_v1_2 import *
def test_enum_ofp_port_state(self):
    self.assertEqual(OFPPS_LINK_DOWN, 1 << 0)
    self.assertEqual(OFPPS_BLOCKED, 1 << 1)
    self.assertEqual(OFPPS_LIVE, 1 << 2)