import unittest
import logging
from os_ken.ofproto.ofproto_v1_2 import *
def test_enum_ofp_vlan_id(self):
    self.assertEqual(OFPVID_PRESENT, 4096)
    self.assertEqual(OFPVID_NONE, 0)