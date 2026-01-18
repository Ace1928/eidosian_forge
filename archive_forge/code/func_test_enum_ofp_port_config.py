import unittest
import logging
from os_ken.ofproto.ofproto_v1_2 import *
def test_enum_ofp_port_config(self):
    self.assertEqual(OFPPC_PORT_DOWN, 1 << 0)
    self.assertEqual(OFPPC_NO_RECV, 1 << 2)
    self.assertEqual(OFPPC_NO_FWD, 1 << 5)
    self.assertEqual(OFPPC_NO_PACKET_IN, 1 << 6)