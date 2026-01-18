import unittest
import logging
from os_ken.ofproto.ofproto_v1_2 import *
def test_enum_ofp_error_type(self):
    self.assertEqual(OFPET_HELLO_FAILED, 0)
    self.assertEqual(OFPET_BAD_REQUEST, 1)
    self.assertEqual(OFPET_BAD_ACTION, 2)
    self.assertEqual(OFPET_BAD_INSTRUCTION, 3)
    self.assertEqual(OFPET_BAD_MATCH, 4)
    self.assertEqual(OFPET_FLOW_MOD_FAILED, 5)
    self.assertEqual(OFPET_GROUP_MOD_FAILED, 6)
    self.assertEqual(OFPET_PORT_MOD_FAILED, 7)
    self.assertEqual(OFPET_TABLE_MOD_FAILED, 8)
    self.assertEqual(OFPET_QUEUE_OP_FAILED, 9)
    self.assertEqual(OFPET_SWITCH_CONFIG_FAILED, 10)
    self.assertEqual(OFPET_ROLE_REQUEST_FAILED, 11)
    self.assertEqual(OFPET_EXPERIMENTER, 65535)