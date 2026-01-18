import unittest
import logging
from os_ken.ofproto.ofproto_v1_2 import *
def test_enum_ofp_switch_config_failed_code(self):
    self.assertEqual(OFPSCFC_BAD_FLAGS, 0)
    self.assertEqual(OFPSCFC_BAD_LEN, 1)
    self.assertEqual(OFPSCFC_EPERM, 2)