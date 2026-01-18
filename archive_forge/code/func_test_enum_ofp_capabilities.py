import unittest
import logging
from os_ken.ofproto.ofproto_v1_2 import *
def test_enum_ofp_capabilities(self):
    self.assertEqual(OFPC_FLOW_STATS, 1 << 0)
    self.assertEqual(OFPC_TABLE_STATS, 1 << 1)
    self.assertEqual(OFPC_PORT_STATS, 1 << 2)
    self.assertEqual(OFPC_GROUP_STATS, 1 << 3)
    self.assertEqual(OFPC_IP_REASM, 1 << 5)
    self.assertEqual(OFPC_QUEUE_STATS, 1 << 6)
    self.assertEqual(OFPC_PORT_BLOCKED, 1 << 8)