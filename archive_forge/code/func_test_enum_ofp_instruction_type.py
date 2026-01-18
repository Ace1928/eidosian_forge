import unittest
import logging
from os_ken.ofproto.ofproto_v1_2 import *
def test_enum_ofp_instruction_type(self):
    self.assertEqual(OFPIT_GOTO_TABLE, 1)
    self.assertEqual(OFPIT_WRITE_METADATA, 2)
    self.assertEqual(OFPIT_WRITE_ACTIONS, 3)
    self.assertEqual(OFPIT_APPLY_ACTIONS, 4)
    self.assertEqual(OFPIT_CLEAR_ACTIONS, 5)
    self.assertEqual(OFPIT_EXPERIMENTER, 65535)