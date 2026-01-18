import unittest
import logging
from os_ken.ofproto.ofproto_v1_2 import *
def test_enum_ofp_config_flags(self):
    self.assertEqual(OFPC_FRAG_NORMAL, 0)
    self.assertEqual(OFPC_FRAG_DROP, 1 << 0)
    self.assertEqual(OFPC_FRAG_REASM, 1 << 1)
    self.assertEqual(OFPC_FRAG_MASK, 3)
    self.assertEqual(OFPC_INVALID_TTL_TO_CONTROLLER, 1 << 2)